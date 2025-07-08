//
// Created by mingqi on 25-7-8.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "gti.h"
#include "objects.h"
#include "neighbor.h"
#include <vector>
#include <unordered_map>
#include <iomanip>

namespace py = pybind11;

class GTIWrapper {
private:
    GTI* gti;
    Objects* data;
    std::unordered_map<int, int> external_to_internal;  // 外部id到内部索引的映射
    std::vector<int> internal_to_external;  // 内部索引到外部id的映射
    std::vector<bool> deleted_flags;  // 标记已删除的向量

public:
    GTIWrapper() : gti(nullptr), data(nullptr) {}

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
        data->vecs.reserve(max_pts);

        external_to_internal.clear();
        internal_to_external.clear();
        deleted_flags.clear();
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

        // 初始化数据结构
        data->dim = dim;
        data->num = n;
        data->type = 0;
        data->vecs.clear();
        data->vecs.reserve(n);

        // 清空映射
        external_to_internal.clear();
        internal_to_external.clear();
        deleted_flags.clear();

        // 添加向量和建立映射
        for (int i = 0; i < n; i++) {
            std::vector<float> vec(ptr_X + i * dim, ptr_X + (i + 1) * dim);
            data->vecs.push_back(vec);

            int external_id = ptr_ids[i];
            // 内部索引就是在data->vecs中的位置
            external_to_internal[external_id] = i;
            internal_to_external.push_back(external_id);
            deleted_flags.push_back(false);
        }

        // 构建GTI索引
        gti->buildGTI(capacity_up_i, capacity_up_l, m, data);

//        if (!data->vecs.empty()) {
//            std::vector<Neighbor> results;
//            float* query_ptr = data->vecs[0].data(); // 取第一个向量作为查询
//            unsigned L = 10; // 可调参数：候选集大小
//            unsigned K = 5;  // 返回前K个结果
//
//            gti->search(query_ptr, L, K, results);
//
//            std::cout << "Search results for the first vector:" << std::endl;
//            for (auto& neighbor : results) {
//                std::cout << "id: " << neighbor.id << ", distance: " << neighbor.dis << std::endl;
//            }
//         }

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

        // 记录插入前的大小
        int old_size = data->vecs.size();

        // 添加向量到主数据结构
        for (int i = 0; i < n; i++) {
            std::vector<float> vec(ptr_X + i * dim, ptr_X + (i + 1) * dim);
//            data->vecs.push_back(vec);
            insert_data->vecs.push_back(vec);

            int external_id = ptr_ids[i];
            int internal_index = old_size + i;  // 新的内部索引

            external_to_internal[external_id] = internal_index;
            internal_to_external.push_back(external_id);
            deleted_flags.push_back(false);
        }

        // 更新data的数量
//        data->num = data->vecs.size();

        // 插入到GTI
        gti->insertGTI(insert_data);

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
        delete_data->num = 0;
        delete_data->type = 0;
        delete_data->vecs.clear();

        // 根据外部id找到对应的向量
        for (int i = 0; i < n; i++) {
            int external_id = ptr_ids[i];
            auto it = external_to_internal.find(external_id);
            if (it != external_to_internal.end()) {
                int internal_index = it->second;
                if (internal_index < data->vecs.size() && !deleted_flags[internal_index]) {
                    delete_data->vecs.push_back(data->vecs[internal_index]);
                    deleted_flags[internal_index] = true;
                    delete_data->num++;
                }
            }
        }

//        for (int i = 0; i < delete_data->num; ++i) {
//            std::cout << "Original: ";
//            for (float val : data->vecs[external_to_internal[ptr_ids[i]]]) {
//                std::cout << std::fixed << std::setprecision(4) << val << " ";
//            }
//            std::cout << "\nCopied:   ";
//            for (float val : delete_data->vecs[i]) {
//                std::cout << std::fixed << std::setprecision(4) << val << " ";
//            }
//            std::cout << "\n----\n";
//        }

        // 从GTI删除
        if (delete_data->num > 0) {
            gti->deleteGTI(delete_data);
        }

        delete delete_data;
    }

    std::pair<py::array_t<int>, py::array_t<float>> query(py::array_t<float> X, int k, int L, bool debug = false) {
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


            // 将内部id转换为外部id
            for (int j = 0; j < k && j < query_results.size(); j++) {
                int internal_id = query_results[j].id;
                int external_id = -1;

                // 检查内部ID是否有效且未被删除
                if (internal_id >= 0 && internal_id < internal_to_external.size() &&
                    !deleted_flags[internal_id]) {
                    external_id = internal_to_external[internal_id];
                }

                ptr_results[i * k + j] = external_id;
                ptr_distances[i * k + j] = query_results[j].dis;
            }

            // 填充剩余位置
            for (int j = query_results.size(); j < k; j++) {
                ptr_results[i * k + j] = -1;
                ptr_distances[i * k + j] = std::numeric_limits<float>::max();
            }
        }

        return std::make_pair(results, distances);
    }

    int size() const {
        return data ? data->num : 0;
    }

    // 添加调试方法
    void debug_info() {
        std::cout << "Data size: " << (data ? data->num : 0) << std::endl;
        std::cout << "Vectors size: " << (data ? data->vecs.size() : 0) << std::endl;
        std::cout << "External to internal mapping size: " << external_to_internal.size() << std::endl;
        std::cout << "Internal to external mapping size: " << internal_to_external.size() << std::endl;

        for (int i = 0; i < std::min(10, (int)internal_to_external.size()); i++) {
            std::cout << "internal[" << i << "] -> external[" << internal_to_external[i] << "]" << std::endl;
        }
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
        .def("query", &GTIWrapper::query, py::arg("X"), py::arg("k"), py::arg("L"), py::arg("debug") = false)
        .def("size", &GTIWrapper::size)
        .def("debug_info", &GTIWrapper::debug_info);
}