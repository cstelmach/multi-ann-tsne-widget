import warnings
from functools import partial
from types import SimpleNamespace as namespace
from typing import Optional  # pylint: disable=unused-import

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFormLayout

from Orange.data import Table, Domain
from Orange.preprocess import preprocess
from Orange.projection import PCA

from Orange.widgets import gui
from Orange.widgets.settings import SettingProvider, ContextSetting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg

# multi-ann
from Orange.widgets.bakk.additional_files import manifold
#from Orange.projection import manifold


_STEP_SIZE = 25
_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 20


class Task(namespace):
    """Completely determines the t-SNE task spec and intermediate results."""
    data = None             # type: Optional[Table]
    normalize = None        # type: Optional[bool]
    pca_components = None   # type: Optional[int]
    pca_projection = None   # type: Optional[Table]
    perplexity = None       # type: Optional[float]
    neighbor_method = None  # cs type: Optional[string]
    multiscale = None       # type: Optional[bool]
    exaggeration = None     # type: Optional[float]
    initialization = None   # type: Optional[np.ndarray]
    affinities = None       # type: Optional[openTSNE.affinity.Affinities]
    tsne_embedding = None   # type: Optional[manifold.TSNEModel]
    iterations_done = 0     # type: int
    method_param_dict = None # type: Optional[Dict]
    # These attributes need not be set by the widget
    tsne = manifold.MultiANNTSNE             # type: Optional[manifold.TSNE]


def pca_preprocessing(data, n_components, normalize):
    projector = PCA(n_components=n_components, random_state=0)
    if normalize:
        projector.preprocessors += (preprocess.Normalize(),)

    model = projector(data)
    return model(data)


def prepare_tsne_obj(data, perplexity, multiscale, exaggeration,
                    neighbor_method, method_param_dict):
    # type: (Table, float, bool, float) -> manifold.TSNE
    """Automatically determine the best parameters for the given data set."""
    # Compute perplexity settings for multiscale
    n_samples = data.X.shape[0]
    if multiscale:
        perplexity = min((n_samples - 1) / 3, 50), min((n_samples - 1) / 3, 500)
    else:
        perplexity = perplexity

    # cs
    neighbor_indextokey = {0:'NNDescent',
                            1:'BallTree',
                            2:'Annoy',
                            3:'NearPy',
                            4:'Hnsw',
                            5:'SW-Graph',
                            6:'NAPP',
                            7:'Brute Force'}

    neighbor_method = neighbor_indextokey[neighbor_method]
    method_param = method_param_dict[neighbor_method]
    neigbhbor_method_plus_param = {'method':neighbor_method,
                                    'param':method_param}

    # Determine whether to use settings for large data sets
    if n_samples > 10_000:
        #neighbor_method, gradient_method = "approx", "fft"
        gradient_method = "fft"
    else:
        #neighbor_method, gradient_method = "exact", "bh"
        gradient_method = "bh"

    # Larger data sets need a larger number of iterations
    if n_samples > 100_000:
        early_exagg_iter, n_iter = 500, 1000
    else:
        early_exagg_iter, n_iter = 250, 750

    return manifold.MultiANNTSNE(
        n_components=2,
        perplexity=perplexity,
        multiscale=multiscale,
        early_exaggeration_iter=early_exagg_iter,
        n_iter=n_iter,
        exaggeration=exaggeration,
        neighbors=neigbhbor_method_plus_param,
        negative_gradient_method=gradient_method,
        theta=0.8,
        random_state=0,
    )


class TSNERunner:
    @staticmethod
    def compute_pca(task, state, **_):
        # Perform PCA preprocessing
        state.set_status("Computing PCA...")
        pca_projection = pca_preprocessing(
            task.data, task.pca_components, task.normalize
        )
        # Apply t-SNE's preprocessors to the data
        task.pca_projection = task.tsne.preprocess(pca_projection)
        state.set_partial_result(("pca_projection", task))

    @staticmethod
    def compute_initialization(task, state, **_):
        # Prepare initial positions for t-SNE
        state.set_status("Preparing initialization...")
        task.initialization = task.tsne.compute_initialization(task.pca_projection.X)
        state.set_partial_result(("initialization", task))

    @staticmethod
    def compute_affinities(task, state, **_):
        # Compute affinities
        state.set_status("Finding nearest neighbors...")
        task.affinities = task.tsne.compute_affinities(task.pca_projection.X)
        state.set_partial_result(("affinities", task))

    @staticmethod
    def compute_tsne(task, state, progress_callback=None):
        tsne = task.tsne

        state.set_status("Running optimization...")

        # If this the first time we're computing t-SNE (otherwise we may just
        # be resuming optimization), we have to assemble the tsne object
        if task.tsne_embedding is None:
            # Assemble a t-SNE embedding object and convert it to a TSNEModel
            task.tsne_embedding = tsne.prepare_embedding(
                task.affinities, task.initialization
            )
            task.tsne_embedding = tsne.convert_embedding_to_model(
                task.pca_projection, task.tsne_embedding
            )
            state.set_partial_result(("tsne_embedding", task))

            if state.is_interruption_requested():
                return

        total_iterations_needed = tsne.early_exaggeration_iter + tsne.n_iter

        def run_optimization(tsne_params: dict, iterations_needed: int) -> bool:
            """Run t-SNE optimization phase. Return value indicates whether or
            not the optimization was interrupted."""
            while task.iterations_done < iterations_needed:
                # Step size can't be larger than the remaining number of iterations
                step_size = min(_STEP_SIZE, iterations_needed - task.iterations_done)
                task.tsne_embedding = task.tsne_embedding.optimize(
                    step_size, **tsne_params
                )
                task.iterations_done += step_size
                state.set_partial_result(("tsne_embedding", task))
                if progress_callback is not None:
                    # The current iterations must be divided by the total
                    # number of iterations, not the number of iterations in the
                    # current phase (iterations_needed)
                    progress_callback(task.iterations_done / total_iterations_needed)

                if state.is_interruption_requested():
                    return True

        # Run early exaggeration phase
        was_interrupted = run_optimization(
            dict(exaggeration=tsne.early_exaggeration, momentum=0.5, inplace=False),
            iterations_needed=tsne.early_exaggeration_iter,
        )
        if was_interrupted:
            return
        # Run regular optimization phase
        run_optimization(
            dict(exaggeration=tsne.exaggeration, momentum=0.8, inplace=False),
            iterations_needed=total_iterations_needed,
        )

    @classmethod
    def run(cls, task, state):
        # type: (Task, TaskState) -> Task

        # Assign weights to each job indicating how much time will be spent on each
        weights = {"pca": 1, "init": 1, "aff": 23, "tsne": 75}
        total_weight = sum(weights.values())

        # Prepare the tsne object and add it to the spec
        task.tsne = prepare_tsne_obj(
            task.data, task.perplexity, task.multiscale, task.exaggeration,
            task.neighbor_method, task.method_param_dict
        )

        job_queue = []
        # Add the tasks that still need to be run to the job queue
        if task.pca_projection is None:
            job_queue.append((cls.compute_pca, weights["pca"]))

        if task.initialization is None:
            job_queue.append((cls.compute_initialization, weights["init"]))

        if task.affinities is None:
            job_queue.append((cls.compute_affinities, weights["aff"]))

        total_iterations = task.tsne.early_exaggeration_iter + task.tsne.n_iter
        if task.tsne_embedding is None or task.iterations_done < total_iterations:
            job_queue.append((cls.compute_tsne, weights["tsne"]))

        job_queue = [(partial(f, task, state), w) for f, w in job_queue]

        # Figure out the total weight of the jobs
        job_weight = sum(j[1] for j in job_queue)
        progress_done = total_weight - job_weight
        for job, job_weight in job_queue:

            def _progress_callback(val):
                state.set_progress_value(
                    (progress_done + val * job_weight) / total_weight * 100
                )

            if state.is_interruption_requested():
                return task

            # Execute the job
            job(progress_callback=_progress_callback)
            # Update the progress bar according to the weights assigned to
            # each job
            progress_done += job_weight
            state.set_progress_value(progress_done / total_weight * 100)

        return task


class OWtSNEGraph(OWScatterPlotBase):
    def update_coordinates(self):
        super().update_coordinates()
        if self.scatterplot_item is not None:
            self.view_box.setAspectLocked(True, 1)


class invalidated:
    # pylint: disable=invalid-name
    pca_projection = affinities = tsne_embedding = False

    def __set__(self, instance, value):
        # `self._invalidate = True` should invalidate everything
        self.pca_projection = self.affinities = self.tsne_embedding = value

    def __bool__(self):
        # If any of the values are invalidated, this should return true
        return self.pca_projection or self.affinities or self.tsne_embedding

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(
            "=".join([k, str(getattr(self, k))])
            for k in ["pca_projection", "affinities", "tsne_embedding"]
        ))


class OWtSNE(OWDataProjectionWidget, ConcurrentWidgetMixin):
    name = "ANN t-SNE"
    description = "Two-dimensional data projection with t-SNE."
    icon = "icons/TSNE.svg"
    priority = 920
    keywords = ["tsne"]

    settings_version = 4
    perplexity = ContextSetting(30)
    multiscale = ContextSetting(False)
    exaggeration = ContextSetting(2)
    pca_components = ContextSetting(_DEFAULT_PCA_COMPONENTS)
    normalize = ContextSetting(True)

    GRAPH_CLASS = OWtSNEGraph
    graph = SettingProvider(OWtSNEGraph)
    embedding_variables_names = ("t-SNE-x", "t-SNE-y")

    left_side_scrolling = True

    # cs
    neighbor_method_index = ["NNDescent", "BallTree", "Annoy", "NearPy", "Hnsw",
    "SW-Graph", "NAPP", "Brute Force"] #"exact", "approx"]
    neighbor_method = 7



    # parameters
    #merging = ContextSetting(0)

    #: Kernel types<
    #Linear, Poly, RBF, Sigmoid = range(4)
    #: Selected kernel type
    #kernel_type = ContextSetting(RBF)
    #: kernel degree
    #degree = ContextSetting(3)
    #: gamma
    #gamma = ContextSetting(0.0)
    #: coef0 (adative constant)
    #coef0 = ContextSetting(0.0)

    search_k = ContextSetting(-1)
    n_trees = ContextSetting(0)
    n_bits = ContextSetting(0)
    hash_counts = ContextSetting(0)
    hnsw_efC = ContextSetting(0)
    M = ContextSetting(0)
    post = ContextSetting(0)
    hnsw_efS = ContextSetting(0)
    swg_efC = ContextSetting(0)
    NN = ContextSetting(0)
    swg_efS = ContextSetting(0)
    numPivot = ContextSetting(0)
    numPivotIndex = ContextSetting(0)
    hashTrickDim = ContextSetting(0)

    _default_value = "default"
    _auto_value = "auto"
    # kernels = (("Linear", "x⋅y"),
    #            ("Polynomial", "(g x⋅y + c)<sup>d</sup>"),
    #            ("RBF", "exp(-g|x-y|²)"),
    #            ("Sigmoid", "tanh(g x⋅y + c)"))

    # parameter dict
    # param_dict = {'search_k' = -1,
    # 'n_bits' = 20, 'hash_counts' = 20,
    # 'M' = 15, 'efConstruction' = 100, 'efSearch' = 100,
    # }
    #search_k = ContextSetting(-1)
    method_param_dict = None

    # Use `invalidated` descriptor so we don't break the usage of
    # `_invalidated` in `OWDataProjectionWidget`, but still allow finer control
    # over which parts of the embedding to invalidate
    _invalidated = invalidated()

    class Information(OWDataProjectionWidget.Information):
        modified = Msg("The parameter settings have been changed. Press "
                       "\"Start\" to rerun with the new settings.")

    class Error(OWDataProjectionWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        constant_data = Msg("Input data is constant")
        no_attributes = Msg("Data has no attributes")
        no_valid_data = Msg("No projection due to no valid data")

    def __init__(self):
        OWDataProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.pca_projection = None  # type: Optional[Table]
        self.initialization = None  # type: Optional[np.ndarray]
        self.affinities = None      # type: Optional[openTSNE.affinity.Affinities]
        self.tsne_embedding = None  # type: Optional[manifold.TSNEModel]
        self.iterations_done = 0    # type: int

    def _add_controls(self):
        self._add_controls_start_box()
        super()._add_controls()

    def _add_controls_start_box(self):
        box = gui.vBox(self.controlArea, True)
        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10,
        )

        self.perplexity_spin = gui.spin(
            box, self, "perplexity", 1, 500, step=1, alignment=Qt.AlignRight,
            callback=self._invalidate_affinities,
        )
        self.controls.perplexity.setDisabled(self.multiscale)
        form.addRow("Perplexity:", self.perplexity_spin)
        form.addRow(gui.checkBox(
            box, self, "multiscale", label="Preserve global structure",
            callback=self._multiscale_changed,
        ))

        sbe = gui.hBox(self.controlArea, False, addToLayout=False)
        gui.hSlider(
            sbe, self, "exaggeration", minValue=1, maxValue=4, step=1,
            callback=self._invalidate_tsne_embedding,
        )
        form.addRow("Exaggeration:", sbe)

        sbp = gui.hBox(self.controlArea, False, addToLayout=False)
        gui.hSlider(
            sbp, self, "pca_components", minValue=2, maxValue=_MAX_PCA_COMPONENTS,
            step=1, callback=self._invalidate_pca_projection,
        )
        form.addRow("PCA components:", sbp)

        self.normalize_cbx = gui.checkBox(
            box, self, "normalize", "Normalize data",
            callback=self._invalidate_pca_projection,
        )
        form.addRow(self.normalize_cbx)

        box.layout().addLayout(form)

        gui.separator(box, 10)
        self.run_button = gui.button(box, self, "Start", callback=self._toggle_run)

        self.para_box = []
        self._add_ann_box()
        self.para_box[0].hide()
    #####

    def _add_ann_box(self):
        # Initialize with the widest label to measure max width
        #self.ann_eq = self.neighbor_method_index[-1]#[1]

        box = gui.hBox(self.controlArea, "Approximate Nearest Neighbor Algorithm")

        # Method dropdown
        self.method_combo = gui.comboBox(
            box, self, "neighbor_method", orientation=Qt.Horizontal,
             items=[i for i in self.neighbor_method_index],
            callback=self._on_ann_changed)#, sendSelectedValue=True)

        parambox = gui.vBox(box)
        common = dict(orientation=Qt.Horizontal, callback=self._invalidate_affinities,
                      alignment=Qt.AlignRight, controlWidth=80, spinType="int")
        spbox = gui.hBox(parambox)
        #gui.rubber(spbox)
        inbox = gui.vBox(spbox)
        self.para_box.append(inbox)
        # Annoy
        search_k_spin = gui.spin(
            inbox, self, "search_k", -10000, 10000, step=1, label=" sk: ", **common)
        search_k_spin.setSpecialValueText(self._default_value)
        n_trees_spin = gui.spin(
            inbox, self, "n_trees", 0, 10000, step=1, label=" nt: ", **common)
        n_trees_spin.setSpecialValueText(self._auto_value)

        # NearPy
        n_bits_spin = gui.spin(
            inbox, self, "n_bits", 0, 10000, step=1, label=" nb: ", **common)
        n_bits_spin.setSpecialValueText(self._default_value)
        hash_counts_spin = gui.spin(
            inbox, self, "hash_counts", 0, 10000, step=1, label=" hc: ", **common)
        hash_counts_spin.setSpecialValueText(self._default_value)

        # Hnsw
        hnsw_efC_spin = gui.spin(
            inbox, self, "hnsw_efC", 0, 3000, step=1, label=" efC: ", **common)
        hnsw_efC_spin.setSpecialValueText(self._auto_value)
        M_spin = gui.spin(
            inbox, self, "M", 0, 200, step=1, label=" M: ", **common)
        M_spin.setSpecialValueText(self._auto_value)
        post_spin = gui.spin(
            inbox, self, "post", 0, 2, step=1, label=" post: ", **common)
        post_spin.setSpecialValueText(self._default_value)
        hnsw_efS_spin = gui.spin(
            inbox, self, "hnsw_efS", 0, 3000, step=1, label=" efS: ", **common)
        hnsw_efS_spin.setSpecialValueText(self._auto_value)

        #SW-Graph
        swg_efC_spin = gui.spin(
            inbox, self, "swg_efC", 0, 3000, step=1, label=" efC: ", **common)
        swg_efC_spin.setSpecialValueText(self._auto_value)
        NN_spin = gui.spin(
            inbox, self, "NN", 0, 200, step=1, label=" NN: ", **common)
        NN_spin.setSpecialValueText(self._auto_value)
        swg_efS_spin = gui.spin(
            inbox, self, "swg_efS", 0, 3000, step=1, label=" efS: ", **common)
        swg_efS_spin.setSpecialValueText(self._auto_value)

        #NAPP
        numPivot_spin = gui.spin(
            inbox, self, "numPivot", 0, 999999, step=1, label=" nP: ", **common)
        numPivot_spin.setSpecialValueText(self._auto_value)
        numPivotIndex_spin = gui.spin(
            inbox, self, "numPivotIndex", 0, 999999, step=1, label=" nPI: ", **common)
        numPivotIndex_spin.setSpecialValueText(self._auto_value)
        hashTrickDim_spin = gui.spin(
            inbox, self, "hashTrickDim", 0, 999999, step=1, label=" hTD: ", **common)
        hashTrickDim_spin.setSpecialValueText(self._default_value)



        # self._kernel_params = [gamma, coef0, degree]
        self._ann_params = [search_k_spin, n_trees_spin, n_bits_spin,
        hash_counts_spin, hnsw_efC_spin, M_spin, post_spin, hnsw_efS_spin,
        swg_efC_spin, NN_spin, swg_efS_spin, numPivot_spin, numPivotIndex_spin,
        hashTrickDim_spin]
        #gui.rubber(parambox)
        # print(self.method_param_dict)
        # method_param_dict = self.get_method_param_dict(self._ann_params)
        # This is the maximal height (all double spins are visible)
        # and the maximal width (the label is initialized to the widest one)
        box.layout().activate()
        # box.setFixedHeight(box.height(100))#sizeHint().height())
        box.setMinimumWidth(box.sizeHint().width())

    def _show_right_kernel(self):
        enabled = [[False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False], # NNDescent
                    [False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False], # BallTree
                    [True, True, False, False, False, False, False,
                    False, False, False, False, False, False, False], # Annoy
                    [False, False, True, True, False, False, False,
                    False, False, False, False, False, False, False], # NearPy
                    [False, False, False, False, True, True, True,
                    True, False, False, False, False, False, False], # Hnsw
                    [False, False, False, False, False, False, False,
                    False, True, True, True, False, False, False], # SW-Graph
                    [False, False, False, False, False, False, False,
                    False, False, False, False, True, True, True], # NAPP
                    [False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False] #Brute Foce
                    ]

        self.ann_eq = self.neighbor_method_index[self.neighbor_method]#[1]
        mask = enabled[self.neighbor_method]
        for spin, enabled in zip(self._ann_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def update_model(self):
        super().update_model()
        sv = None
        if self.model is not None:
            sv = self.data[self.model.skl_model.support_]
        self.Outputs.support_vectors.send(sv)

    def _on_ann_changed(self):
        if self.neighbor_method != 0:# or self.neighbor_method =='Annoy':
            self.para_box[0].show()
        self._show_right_kernel()
        self._invalidate_affinities()
        #self.settings_changed()

    # def get_method_param_dict(self, ap):
    #     #[search_k, n_bits, hash_counts, hnsw_efC, M, post,
    #     #hnsw_efS, swg_efC, NN, swg_efS, numPivot, numPivotIndex]
    #     method_param_dict = {'NNDescent': False,
    #     'BallTree': False,
    #     'Annoy': {'search_k': ap[0]},
    #     'NearPy':{'nb':ap[1], 'hash_counts':ap[2]},
    #     'Hnsw':{'index_param':{'M':ap[3], 'efConstruction': ap[4], 'post':ap[5]} ,
    #             'search_param':{'efSearch':ap[6]}},
    #     'SW-Graph':{'index_param':{'NN':ap[7], 'efConstruction': ap[8]} ,
    #             'search_param':{'efSearch':ap[9]}},
    #     'NAPP': {'index_param':{'numPivot':ap[10], 'numPivotIndex': ap[11]},
    #     'Brute Force': False}}
    #     return method_param_dict


    #####

    def _multiscale_changed(self):
        self.controls.perplexity.setDisabled(self.multiscale)
        self._invalidate_affinities()

    def _invalidate_pca_projection(self):
        self._invalidated.pca_projection = True
        self._invalidate_affinities()

    def _invalidate_affinities(self):
        self._invalidated.affinities = True
        self._invalidate_tsne_embedding()


    def _invalidate_tsne_embedding(self):
        self._invalidated.tsne_embedding = True
        self._stop_running_task()
        self._set_modified(True)

    def _stop_running_task(self):
        self.cancel()
        self.run_button.setText("Start")

    def _set_modified(self, state):
        """Mark the widget (GUI) as containing modified state."""
        if self.data is None:
            # Does not apply when we have no data
            state = False
        self.Information.modified(shown=state)

    def check_data(self):
        def error(err):
            err()
            self.data = None

        # `super().check_data()` clears all messages so we have to remember if
        # it was shown
        # pylint: disable=assignment-from-no-return
        should_show_modified_message = self.Information.modified.is_shown()
        super().check_data()

        if self.data is None:
            return

        self.Information.modified(shown=should_show_modified_message)

        if len(self.data) < 2:
            error(self.Error.not_enough_rows)

        elif not self.data.domain.attributes:
            error(self.Error.no_attributes)

        elif not self.data.is_sparse():
            if np.all(~np.isfinite(self.data.X)):
                error(self.Error.no_valid_data)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Degrees of freedom .*", RuntimeWarning)
                    if np.nan_to_num(np.nanstd(self.data.X, axis=0)).sum() \
                            == 0:
                        error(self.Error.constant_data)

    def get_embedding(self):
        if self.tsne_embedding is None:
            self.valid_data = None
            return None

        embedding = self.tsne_embedding.embedding.X
        self.valid_data = np.ones(len(embedding), dtype=bool)
        return embedding

    def _toggle_run(self):
        # If no data, there's nothing to do
        if self.data is None:
            return

        # Pause task
        if self.task is not None:
            self.cancel()
            self.run_button.setText("Resume")
            self.commit()
        # Resume task
        else:
            self.run()

    def handleNewSignals(self):
        # We don't bother with the granular invalidation flags because
        # `super().handleNewSignals` will just set all of them to False or will
        # do nothing. However, it's important we remember its state because we
        # won't call `run` if needed. `run` also relies on the state of
        # `_invalidated` to properly set the intermediate values to None
        prev_invalidated = bool(self._invalidated)
        super().handleNewSignals()
        self._invalidated = prev_invalidated

        if self._invalidated:
            self.run()

    def init_attr_values(self):
        super().init_attr_values()

        if self.data is not None:
            n_attrs = len(self.data.domain.attributes)
            max_components = min(_MAX_PCA_COMPONENTS, n_attrs)
        else:
            max_components = _MAX_PCA_COMPONENTS

        # We set this to the default number of components here so it resets
        # properly, any previous settings will be restored from context
        # settings a little later
        self.controls.pca_components.setMaximum(max_components)
        self.controls.pca_components.setValue(_DEFAULT_PCA_COMPONENTS)

        self.exaggeration = 1

    def enable_controls(self):
        super().enable_controls()

        if self.data is not None:
            # PCA doesn't support normalization on sparse data, as this would
            # require centering and normalizing the matrix
            self.normalize_cbx.setDisabled(self.data.is_sparse())
            if self.data.is_sparse():
                self.normalize = False
                self.normalize_cbx.setToolTip(
                    "Data normalization is not supported on sparse matrices."
                )
            else:
                self.normalize_cbx.setToolTip("")

        # Disable the perplexity spin box if multiscale is turned on
        self.controls.perplexity.setDisabled(self.multiscale)

    def run(self):
        # Reset invalidated values as indicated by the flags
        if self._invalidated.pca_projection:
            self.pca_projection = None
        if self._invalidated.affinities:
            self.affinities = None
        if self._invalidated.tsne_embedding:
            self.iterations_done = 0
            self.tsne_embedding = None

        self._set_modified(False)
        self._invalidated = False

        # When the data is invalid, it is set to `None` and an error is set,
        # therefore it would be erroneous to clear the error here
        if self.data is not None:
            self.run_button.setText("Stop")

        # Cancel current running task
        self.cancel()

        if self.data is None:
            return

        self.method_param_dict = {'NNDescent': False,
        'BallTree': False,
        'Annoy': {'search_k': self.search_k, 'n_trees': self.n_trees},
        'NearPy':{'n_bits':self.n_bits, 'hash_counts':self.hash_counts},
        'Hnsw':{'M':self.M, 'hnsw_efConstruction': self.hnsw_efC, 'post':self.post,
                'hnsw_efSearch':self.hnsw_efS},
        'SW-Graph':{'NN':self.NN, 'swg_efConstruction':self.swg_efC, 'swg_efSearch':self.swg_efS},
        'NAPP': {'numPivot':self.numPivot, 'numPivotIndex': self.numPivotIndex,
                'hashTrickDim':self.hashTrickDim},
        'Brute Force': False}

        task = Task(
            data=self.data,
            normalize=self.normalize,
            pca_components=self.pca_components,
            pca_projection=self.pca_projection,
            perplexity=self.perplexity,
            neighbor_method=self.neighbor_method, #cs
            multiscale=self.multiscale,
            exaggeration=self.exaggeration,
            initialization=self.initialization,
            affinities=self.affinities,
            tsne_embedding=self.tsne_embedding,
            iterations_done=self.iterations_done,
            method_param_dict=self.method_param_dict
        )
        return self.start(TSNERunner.run, task)

    def __ensure_task_same_for_pca(self, task: Task):
        assert self.data is not None
        assert task.data is self.data
        assert task.normalize == self.normalize
        assert task.pca_components == self.pca_components
        assert isinstance(task.pca_projection, Table) and \
            len(task.pca_projection) == len(self.data)

    def __ensure_task_same_for_initialization(self, task: Task):
        assert isinstance(task.initialization, np.ndarray) and \
            len(task.initialization) == len(self.data)

    def __ensure_task_same_for_affinities(self, task: Task):
        assert task.perplexity == self.perplexity
        #assert task.neighbor_method == self.neighbor_method #cs
        assert task.multiscale == self.multiscale

    def __ensure_task_same_for_embedding(self, task: Task):
        assert task.exaggeration == self.exaggeration
        assert isinstance(task.tsne_embedding, manifold.TSNEModel) and \
            len(task.tsne_embedding.embedding) == len(self.data)

    def on_partial_result(self, value):
        # type: (Tuple[str, Task]) -> None
        which, task = value

        if which == "pca_projection":
            self.__ensure_task_same_for_pca(task)
            self.pca_projection = task.pca_projection
        elif which == "initialization":
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_initialization(task)
            self.initialization = task.initialization
        elif which == "affinities":
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_affinities(task)
            self.affinities = task.affinities
        elif which == "tsne_embedding":
            self.__ensure_task_same_for_pca(task)
            self.__ensure_task_same_for_initialization(task)
            self.__ensure_task_same_for_affinities(task)
            self.__ensure_task_same_for_embedding(task)

            prev_embedding, self.tsne_embedding = self.tsne_embedding, task.tsne_embedding
            self.iterations_done = task.iterations_done
            # If this is the first partial result we've gotten, we've got to
            # setup the plot
            if prev_embedding is None:
                self.setup_plot()
            # Otherwise, just update the point positions
            else:
                self.graph.update_coordinates()
                self.graph.update_density()
        else:
            raise RuntimeError(
                "Unrecognized partial result called with `%s`" % which
            )

    def on_done(self, task):
        # type: (Task) -> None
        self.run_button.setText("Start")
        # NOTE: All of these have already been set by on_partial_result,
        # we double check that they are aliases
        if task.pca_projection is not None:
            self.__ensure_task_same_for_pca(task)
            assert task.pca_projection is self.pca_projection
        if task.initialization is not None:
            self.__ensure_task_same_for_initialization(task)
            assert task.initialization is self.initialization
        if task.affinities is not None:
            assert task.affinities is self.affinities
        if task.tsne_embedding is not None:
            self.__ensure_task_same_for_embedding(task)
            assert task.tsne_embedding is self.tsne_embedding

        self.commit()

    def _get_projection_data(self):
        if self.data is None:
            return None

        data = self.data.transform(
            Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + self._get_projection_variables()
            )
        )
        data.metas[:, -2:] = self.get_embedding()
        if self.tsne_embedding is not None:
            data.domain = Domain(
                self.data.domain.attributes,
                self.data.domain.class_vars,
                self.data.domain.metas + self.tsne_embedding.domain.attributes,
            )
        return data

    def clear(self):
        """Clear widget state. Note that this doesn't clear the data."""
        super().clear()
        self.run_button.setText("Start")
        self.cancel()
        self.pca_projection = None
        self.initialization = None
        self.affinities = None
        self.tsne_embedding = None
        self.iterations_done = 0

    def onDeleteWidget(self):
        self.clear()
        self.data = None
        self.shutdown()
        super().onDeleteWidget()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 3:
            if "selection_indices" in settings:
                settings["selection"] = settings["selection_indices"]
        if version < 4:
            settings.pop("max_iter", None)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 3:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


if __name__ == "__main__":
    import sys
    data = Table(sys.argv[1] if len(sys.argv) > 1 else "iris")
    WidgetPreview(OWtSNE).run(
        set_data=data,
        set_subset_data=data[np.random.choice(len(data), 10)],
    )
