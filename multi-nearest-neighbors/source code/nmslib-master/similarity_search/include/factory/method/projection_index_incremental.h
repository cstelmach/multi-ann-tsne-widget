/**
 * Non-metric Space Library
 *
 * Main developers: Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, Ben Frederickson, David Novak
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib
 *
 * Copyright (c) 2013-2018
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
#ifndef _FACTORY_PROJ_INDEX_INCR_H_
#define _FACTORY_PROJ_INDEX_INCR_H_

#include <method/projection_index_incremental.h>

namespace similarity {

/*
 * Creating functions.
 */

template <typename dist_t>
Index<dist_t>* CreateProjectionIndexIncremental(bool PrintProgress,
                           const string& SpaceType,
                           Space<dist_t>& space,
                           const ObjectVector& DataObjects) {
  return new ProjectionIndexIncremental<dist_t>(PrintProgress,
                                                space,
                                                DataObjects);

}

/*
 * End of creating functions.
 */

}

#endif
