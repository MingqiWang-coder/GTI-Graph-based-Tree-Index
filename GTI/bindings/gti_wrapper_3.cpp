#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <cstring>
#include "process.h"
namespace py = pybind11;

class GTIWrapper {
public:
    GTIWrapper();
    ~GTIWrapper();

    // Setup the GTI index
    void setup(int max_pts, int ndim, unsigned capacity_up_i, unsigned capacity_up_l, int m);

    // Build the index with initial data
    void build(const float* data, const int* ids, int n_points, unsigned capacity_up_i, unsigned capacity_up_l, int m);

    // Insert new vectors
    void insert(const float* data, const int* ids, int n_points);

    // Remove vectors by IDs
    void remove(const int* ids, int n_ids);

    // Query k nearest neighbors
    void query(const float* queries, int n_queries, int k, unsigned l,
              int* results, float* distances);

    // Get memory usage
    size_t size() const;

    // Check if index is built
    bool is_built() const;

private:
    GTI* gti;
    Objects* data_objects;
    Objects* data_;
    int max_points;
    int dimension;
    bool built;

    // ID mapping: external_id -> internal_index, internal_index -> external_id
    std::unordered_map<int, int> id_to_index;
    std::unordered_map<int, int> index_to_id;
    int next_internal_index;

    // Helper functions
    void create_objects_from_data(const float* data, const int* ids, int n_points, Objects* objects);
    void update_id_mapping(const int* ids, int n_points, int start_index);
    void remove_from_id_mapping(int external_id);
    void cleanup();
};

// Implementation

GTIWrapper::GTIWrapper() : gti(nullptr), data_objects(nullptr), max_points(0), dimension(0), data_(nullptr),
                          built(false), next_internal_index(0) {
}

GTIWrapper::~GTIWrapper() {
    cleanup();
}

void GTIWrapper::setup(int max_pts, int ndim, unsigned capacity_up_i, unsigned capacity_up_l, int m) {
    max_points = max_pts;
    dimension = ndim;

    // Initialize GTI
    if (gti) {
        delete gti;
    }
    gti = new GTI();

    // Initialize data objects container
    if (data_objects) {
        data_objects->release();
        delete data_objects;
    }
    data_objects = new Objects();
    data_objects->dim = dimension;
    data_objects->type = 0; // L2 distance

    data_ = new Objects();
    data_->dim = dimension;
    data_->type = 0; // L2 distance

    // Clear ID mappings
    id_to_index.clear();
    index_to_id.clear();
    next_internal_index = 0;

    built = false;
    std::cout << "GTI wrapper setup completed: max_pts=" << max_pts
              << ", ndim=" << ndim << ", capacity_up_i=" << capacity_up_i
              << ", capacity_up_l=" << capacity_up_l << ", m=" << m << std::endl;
}

void GTIWrapper::build(const float* data, const int* ids, int n_points, unsigned capacity_up_i, unsigned capacity_up_l, int m) {
    if (!gti || !data_objects) {
        std::cerr << "Error: GTI not properly initialized. Call setup() first." << std::endl;
        return;
    }

    // Create Objects from input data
    create_objects_from_data(data, ids, n_points, data_objects);
    create_objects_from_data(data, ids, n_points, data_);

    // Update ID mappings
    update_id_mapping(ids, n_points, 0);

    // Build the GTI index
    std::cout << "Building GTI index with " << n_points << " points..." << std::endl;
    gti->buildGTI(capacity_up_i, capacity_up_l, m, data_);
    built = true;


    gti->getTreeSize();
    double sizeInMB = gti->tree_size / (1024.0 * 1024.0);
    std::cout << "Size of tree: " << sizeInMB << std::endl;
}

void GTIWrapper::insert(const float* data, const int* ids, int n_points) {
    if (!built) {
        std::cerr << "Error: Index must be built before inserting" << std::endl;
        return;
    }

    // Create Objects for new data
//    Objects* new_objects = new Objects();
    auto new_objects = std::make_unique<Objects>();
    create_objects_from_data(data, ids, n_points, new_objects.get());

    // Insert into GTI
    gti->insertGTI(new_objects.get());

    // Update data_objects with new data and ID mappings
    int start_index = data_objects->num;
    for (int i = 0; i < n_points; i++) {
        data_objects->vecs.push_back(new_objects->vecs[i]);
    }
    data_objects->num += n_points;

    // Update ID mappings for new data
    update_id_mapping(ids, n_points, start_index);

    gti->getTreeSize();
    double sizeInMB = gti->tree_size / (1024.0 * 1024.0);
    std::cout << "Size of tree: " << sizeInMB << std::endl;
    // Clean up temporary objects
//    new_objects->release();
//    delete new_objects;
}

void GTIWrapper::remove(const int* ids, int n_ids) {
    if (!built) {
        std::cerr << "Error: Index must be built before removing" << std::endl;
        return;
    }

    // Create Objects for data to be removed
//    Objects* delete_objects = new Objects();
    auto delete_objects = std::make_unique<Objects>();
    delete_objects->dim = dimension;
    delete_objects->type = 0;
    delete_objects->num = 0;

    std::vector<int> found_indices;

    // Find vectors to delete using ID mapping
    for (int i = 0; i < n_ids; i++) {
        int external_id = ids[i];
        auto it = id_to_index.find(external_id);

        if (it != id_to_index.end()) {
            int internal_index = it->second;
            if (internal_index >= 0 && internal_index < static_cast<int>(data_objects->vecs.size())) {
                delete_objects->vecs.push_back(data_objects->vecs[internal_index]);
                found_indices.push_back(internal_index);
            } else {
                std::cerr << "Warning: Invalid internal index " << internal_index
                         << " for external ID " << external_id << std::endl;
            }
        } else {
            std::cerr << "Warning: External ID " << external_id << " not found in mapping" << std::endl;
        }
    }

    if (delete_objects->vecs.empty()) {
        std::cerr << "Error: No valid vectors found for deletion" << std::endl;
//        delete_objects->release();
//        delete delete_objects;
        return;
    }

    delete_objects->num = delete_objects->vecs.size();

    // Delete from GTI
    gti->deleteGTI(delete_objects.get());

    // Update ID mappings - remove deleted IDs
    for (int i = 0; i < n_ids; i++) {
        int external_id = ids[i];
        auto it = id_to_index.find(external_id);
        if (it != id_to_index.end()) {
            remove_from_id_mapping(external_id);
        }
    }

    // Clean up
//    delete_objects->release();
//    delete delete_objects;
}

void GTIWrapper::query(const float* queries, int n_queries, int k, unsigned l,
                      int* results, float* distances) {
    if (!built) {
        std::cerr << "Error: Index must be built before querying" << std::endl;
        return;
    }

    // Create query objects
//    Objects* query_objects = new Objects();
    auto query_objects = std::make_unique<Objects>();
    query_objects->dim = dimension;
    query_objects->type = 0;
    query_objects->num = n_queries;
    query_objects->vecs.resize(n_queries);

    // Copy query data
    for (int i = 0; i < n_queries; i++) {
        query_objects->vecs[i].resize(dimension);
        std::memcpy(query_objects->vecs[i].data(), queries + i * dimension, dimension * sizeof(float));
    }

    // Perform queries
    NN query_results(n_queries);
    for (int i = 0; i < n_queries; i++) {
        gti->search(query_objects->vecs[i].data(), l, k, query_results[i]);
    }

    // Copy results and convert internal IDs to external IDs
    for (int i = 0; i < n_queries; i++) {
        for (int j = 0; j < k && j < static_cast<int>(query_results[i].size()); j++) {
            int internal_id = query_results[i][j].id;

            // Convert internal ID to external ID using mapping
            auto it = index_to_id.find(internal_id);
            if (it != index_to_id.end()) {
                results[i * k + j] = it->second; // Use external ID
            } else {
                results[i * k + j] = internal_id; // Fallback to internal ID
//                std::cerr << "Warning: Internal ID " << internal_id
//                         << " not found in mapping, using internal ID" << std::endl;
            }

            if (distances) {
                distances[i * k + j] = query_results[i][j].dis;
            }
        }
        // Fill remaining slots with -1 if fewer than k results
        for (int j = query_results[i].size(); j < k; j++) {
            results[i * k + j] = -1;
            if (distances) {
                distances[i * k + j] = -1.0f;
            }
        }
    }

    // Clean up
//    query_objects->release();
//    delete query_objects;
}

size_t GTIWrapper::size() const {
    if (!built || !gti) {
        return 0;
    }
    return gti->tree_size;
}

bool GTIWrapper::is_built() const {
    return built;
}

void GTIWrapper::create_objects_from_data(const float* data, const int* ids, int n_points, Objects* objects) {
    objects->dim = dimension;
    objects->type = 0; // L2 distance
    objects->num = n_points;
    objects->vecs.resize(n_points);

    for (int i = 0; i < n_points; i++) {
        objects->vecs[i].resize(dimension);
        std::memcpy(objects->vecs[i].data(), data + i * dimension, dimension * sizeof(float));
    }
}

void GTIWrapper::update_id_mapping(const int* ids, int n_points, int start_index) {
    for (int i = 0; i < n_points; i++) {
        int external_id = ids[i];
        int internal_index = start_index + i;

        // Update bidirectional mapping
        id_to_index[external_id] = internal_index;
        index_to_id[internal_index] = external_id;

        // Update next internal index
        next_internal_index = std::max(next_internal_index, internal_index + 1);
    }
}

void GTIWrapper::remove_from_id_mapping(int external_id) {
    auto it = id_to_index.find(external_id);
    if (it != id_to_index.end()) {
        int internal_index = it->second;

        // Remove from both mappings
        id_to_index.erase(external_id);
        index_to_id.erase(internal_index);

    }
}

void GTIWrapper::cleanup() {
    if (gti) {
        delete gti;
        gti = nullptr;
    }

    if (data_objects) {
        data_objects->release();
        delete data_objects;
        data_objects = nullptr;
    }

    // Clear ID mappings
    id_to_index.clear();
    index_to_id.clear();
    next_internal_index = 0;

    built = false;
}


PYBIND11_MODULE(gti_wrapper, m) {
    m.doc() = "GTI (Graph-based Tree Index) Python wrapper";

    py::class_<GTIWrapper>(m, "GTIWrapper")
        .def(py::init<>())
        .def("setup", &GTIWrapper::setup,
             "Setup the GTI index",
             py::arg("max_pts"), py::arg("ndim"),
             py::arg("capacity_up_i"), py::arg("capacity_up_l"), py::arg("m"))

        .def("build", [](GTIWrapper& self,
                        py::array_t<float> data,
                        py::array_t<int> ids,
                        unsigned capacity_up_i,
                        unsigned capacity_up_l,
                        int m) {
            auto data_buf = data.request();
            auto ids_buf = ids.request();

            if (data_buf.ndim != 2) {
                throw std::runtime_error("Data must be 2D array");
            }
            if (ids_buf.ndim != 1) {
                throw std::runtime_error("IDs must be 1D array");
            }

            int n_points = data_buf.shape[0];
            if (n_points != ids_buf.shape[0]) {
                throw std::runtime_error("Number of data points and IDs must match");
            }

            self.build(static_cast<float*>(data_buf.ptr),
                      static_cast<int*>(ids_buf.ptr),
                      n_points, capacity_up_i, capacity_up_l, m);
        }, "Build the GTI index",
        py::arg("data"), py::arg("ids"),
        py::arg("capacity_up_i"), py::arg("capacity_up_l"), py::arg("m"))

        .def("insert", [](GTIWrapper& self,
                         py::array_t<float> data,
                         py::array_t<int> ids) {
            auto data_buf = data.request();
            auto ids_buf = ids.request();

            if (data_buf.ndim != 2) {
                throw std::runtime_error("Data must be 2D array");
            }
            if (ids_buf.ndim != 1) {
                throw std::runtime_error("IDs must be 1D array");
            }

            int n_points = data_buf.shape[0];
            if (n_points != ids_buf.shape[0]) {
                throw std::runtime_error("Number of data points and IDs must match");
            }

            self.insert(static_cast<float*>(data_buf.ptr),
                       static_cast<int*>(ids_buf.ptr),
                       n_points);
        }, "Insert vectors into the GTI index",
        py::arg("data"), py::arg("ids"))

        .def("remove", [](GTIWrapper& self, py::array_t<int> ids) {
            auto ids_buf = ids.request();

            if (ids_buf.ndim != 1) {
                throw std::runtime_error("IDs must be 1D array");
            }

            self.remove(static_cast<int*>(ids_buf.ptr),
                       static_cast<int>(ids_buf.shape[0]));
        }, "Remove vectors by IDs",
        py::arg("ids"))

        .def("query", [](GTIWrapper& self,
                        py::array_t<float> queries,
                        int k,
                        unsigned l) -> py::tuple {
            auto queries_buf = queries.request();

            if (queries_buf.ndim != 2) {
                throw std::runtime_error("Queries must be 2D array");
            }

            int n_queries = queries_buf.shape[0];

            // Allocate output arrays
            auto results = py::array_t<int>({n_queries, k});
            auto distances = py::array_t<float>({n_queries, k});

            auto results_buf = results.request();
            auto distances_buf = distances.request();

            self.query(static_cast<float*>(queries_buf.ptr),
                      n_queries, k, l,
                      static_cast<int*>(results_buf.ptr),
                      static_cast<float*>(distances_buf.ptr));

            return py::make_tuple(results, distances);
        }, "Query k nearest neighbors",
        py::arg("queries"), py::arg("k"), py::arg("l"))

        .def("size", &GTIWrapper::size,
             "Get memory usage of the index")

        .def("is_built", &GTIWrapper::is_built,
             "Check if index is built");
}