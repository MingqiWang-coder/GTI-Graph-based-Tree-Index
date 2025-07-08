#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "gti.h"
#include "objects.h"
#include "neighbor.h"
#include <vector>
#include <unordered_map>

namespace py = pybind11;

class GTIWrapper {
private:
    GTI* gti;
    Objects* data;
    std::unordered_map<int, int> external_id_to_internal_index;  // 仅用于删除操作的映射
    std::vector<bool> deleted_flags;  // 标记已删除的向量
    int next_internal_id;

public:
    GTIWrapper() : gti(nullptr), data(nullptr), next_internal_id(0) {}

    ~GTIWrapper() {
        if (gti) delete gti;
        if (data) delete data;
    }

    void setup(int max_pts, int ndim, int capacity_up_i, int capacity_up_l, int m) {
        if (gti) delete gti;
        if (data) delete data;

        gti = new GTI();
        data = new Objects();
        data->dim = ndim;
        data->num = 0;
        data->type = 0;
        data->vecs.clear();
        data->vecs.reserve(max_pts);

        external_id_to_internal_index.clear();
        deleted_flags.clear();
        next_internal_id = 0;
    }

    void build(py::array_t<float> X, py::array_t<int> ids, int capacity_up_i, int capacity_up_l, int m) {
        auto buf_X = X.request();
        auto buf_ids = ids.request();

        if (buf_X.ndim != 2 || buf_ids.ndim != 1) {
            throw std::runtime_error("Input arrays must be 2D (X) and 1D (ids)");
        }

        int n = buf_X.shape[0];
        int dim = buf_X.shape[1];

        if (buf_ids.shape[0] != n) {
            throw std::runtime_error("Number of vectors and ids must match");
        }

        float* ptr_X = static_cast<float*>(buf_X.ptr);
        int* ptr_ids = static_cast<int*>(buf_ids.ptr);

        // 清空映射
        external_id_to_internal_index.clear();
        deleted_flags.clear();
        next_internal_id = 0;

        // 添加向量和建立映射（仅用于删除操作）
        for (int i = 0; i < n; i++) {
            std::vector<float> vec(ptr_X + i * dim, ptr_X + (i + 1) * dim);
            data->vecs.push_back(vec);

            int external_id = ptr_ids[i];
            external_id_to_internal_index[external_id] = i;
            deleted_flags.push_back(false);
        }
        next_internal_id = n;
        data->num = n;
        // 构建GTI索引
        gti->buildGTI(capacity_up_i, capacity_up_l, m, data);

        gti->getTreeSize();
         double sizeInMB = gti->tree_size / (1024.0 * 1024.0);
         std::cout << "Size of tree: " << sizeInMB << std::endl;
    }

    void insert(py::array_t<float> X, py::array_t<int> ids) {
        auto buf_X = X.request();
        auto buf_ids = ids.request();

        if (buf_X.ndim != 2 || buf_ids.ndim != 1) {
            throw std::runtime_error("Input arrays must be 2D (X) and 1D (ids)");
        }

        int n = buf_X.shape[0];
        int dim = buf_X.shape[1];

        if (buf_ids.shape[0] != n) {
            throw std::runtime_error("Number of vectors and ids must match");
        }

        float* ptr_X = static_cast<float*>(buf_X.ptr);
        int* ptr_ids = static_cast<int*>(buf_ids.ptr);

        // 创建插入数据对象
        Objects* insert_data = new Objects();
        insert_data->dim = dim;
        insert_data->num = n;
        insert_data->type = 0;
        insert_data->vecs.clear();
        insert_data->vecs.reserve(n);

        // 添加向量和建立映射（仅用于删除操作）
        for (int i = 0; i < n; i++) {
            std::vector<float> vec(ptr_X + i * dim, ptr_X + (i + 1) * dim);
            insert_data->vecs.push_back(vec);
//            data->vecs.push_back(vec);

            int external_id = ptr_ids[i];
            int internal_index = next_internal_id++;
            external_id_to_internal_index[external_id] = internal_index;

            // 扩展内部数据结构
            if (internal_index >= deleted_flags.size()) {
                deleted_flags.resize(internal_index + 1);
            }
            deleted_flags[internal_index] = false;
        }

        // 插入到GTI
        gti->insertGTI(insert_data);
        gti->getTreeSize();
        double sizeInMB = gti->tree_size / (1024.0 * 1024.0);
        std::cout << "Size of tree: " << sizeInMB << "num of data：" << data->num << std::endl;
        delete insert_data;
    }

    void remove(py::array_t<int> ids) {
        auto buf_ids = ids.request();

        if (buf_ids.ndim != 1) {
            throw std::runtime_error("IDs array must be 1D");
        }

        int n = buf_ids.shape[0];
        int* ptr_ids = static_cast<int*>(buf_ids.ptr);

        // 创建删除数据对象
        Objects* delete_data = new Objects();
        delete_data->dim = data->dim;
        delete_data->num = n;
        delete_data->type = 0;
        delete_data->vecs.clear();
        delete_data->vecs.reserve(n);

        // 根据外部id找到对应的向量
        for (int i = 0; i < n; i++) {
            int external_id = ptr_ids[i];
            auto it = external_id_to_internal_index.find(external_id);
            if (it != external_id_to_internal_index.end()) {
                int internal_index = it->second;
                if (internal_index < data->vecs.size() && !deleted_flags[internal_index]) {
                    delete_data->vecs.push_back(data->vecs[internal_index]);
                    deleted_flags[internal_index] = true;
                }
            }
        }

        delete_data->num = delete_data->vecs.size();

        // 从GTI删除
        if (delete_data->num > 0) {
            gti->deleteGTI(delete_data);
        }

        delete delete_data;
    }

    std::pair<py::array_t<int>, py::array_t<float>> query(py::array_t<float> X, int k, int L) {
        auto buf_X = X.request();

        if (buf_X.ndim != 2) {
            throw std::runtime_error("Query array must be 2D");
        }

        int n = buf_X.shape[0];
        int dim = buf_X.shape[1];
        float* ptr_X = static_cast<float*>(buf_X.ptr);

        // 创建结果数组
        auto results = py::array_t<int>({n, k});
        auto distances = py::array_t<float>({n, k});
        auto buf_results = results.request();
        auto buf_distances = distances.request();
        int* ptr_results = static_cast<int*>(buf_results.ptr);
        float* ptr_distances = static_cast<float*>(buf_distances.ptr);

        // 对每个查询向量进行搜索
        for (int i = 0; i < n; i++) {
            std::vector<Neighbor> query_results;
            float* query_vec = ptr_X + i * dim;

            gti->search(query_vec, L, k, query_results);

            // 直接返回内部id
            for (int j = 0; j < k && j < query_results.size(); j++) {
                int internal_id = query_results[j].id;
                float distance = query_results[j].dis;

                if(j < 3 && i == 0) {  // 只打印第一个查询的前3个结果
                    std::cout << "Result " << j << ": id = " << internal_id
                              << ", distance = " << distance
                              << ",nid = " << query_results[j].nid
                              << ",oid = " << query_results[j].oid << std::endl;
                }

                ptr_results[i * k + j] = internal_id;
                ptr_distances[i * k + j] = distance;
            }

            // 填充剩余位置
//            for (int j = query_results.size(); j < k; j++) {
//                ptr_results[i * k + j] = -1;
//                ptr_distances[i * k + j] = std::numeric_limits<float>::max();
//            }
        }

        return std::make_pair(results, distances);
    }

    int size() const {
        return data ? data->num : 0;
    }
};

PYBIND11_MODULE(gti_wrapper, m) {
    m.doc() = "GTI Python wrapper";

    py::class_<GTIWrapper>(m, "GTIWrapper")
        .def(py::init<>())
        .def("setup", &GTIWrapper::setup)
        .def("build", &GTIWrapper::build)
        .def("insert", &GTIWrapper::insert)
        .def("remove", &GTIWrapper::remove)
        .def("query", &GTIWrapper::query)
        .def("size", &GTIWrapper::size);
}