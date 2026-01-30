#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace lazyllm {

struct NodeGroup {
    enum class Type {
        ORIGINAL, CHUNK, SUMMARY, IMAGE_INFO, QUESTION_ANSWER, OTHER
    };
    std::string parent;
    std::string display_name;
    Type type;
    NodeGroup(
        const std::string& parent,
        const std::string& display_name,
        const Type& type = Type::ORIGINAL) :
            parent(parent), display_name(display_name), type(type) {}
};

class DocumentStore {
public:
    // RAG system metadata keys
    static constexpr std::string_view RAG_KB_ID_KEY = "kb_id";
    static constexpr std::string_view RAG_DOC_ID_KEY = "docid";

    DocumentStore() = delete;
    explicit DocumentStore(
        const pybind11::object &store,
        const std::unordered_map<std::string, NodeGroup> &map) :
            _py_store(store), _node_groups_map(map) {}

    // Cache-aware factory to avoid rebuilding wrappers for the same Python store.
    static std::shared_ptr<DocumentStore> from_store(
        const pybind11::object &store, const std::unordered_map<std::string, NodeGroup> &map) {
        if (store.is_none()) return nullptr;

        pybind11::gil_scoped_acquire gil;
        PyObject *key = store.ptr();
        auto &cache = store_cache();
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (auto existing = it->second.lock())
                return existing;
        }
        auto created = std::shared_ptr<DocumentStore>(new DocumentStore(store, map));
        cache[key] = created;
        return created;
    }

    bool is_group_active(const std::string& grp) const {
        pybind11::gil_scoped_acquire gil;
        pybind11::object fn = _py_store.attr("is_group_active");
        return fn(grp).cast<bool>();
    }

    pybind11::object get_node(
        const std::string& group_name,
        const std::string& uid,
        const std::string& kb_id
    ) const {
        pybind11::gil_scoped_acquire gil;
        pybind11::object fn = _py_store.attr("get_nodes");
        pybind11::object result = fn(
            pybind11::arg("group_name") = group_name,
            pybind11::arg("uids") = std::vector<std::string>({uid}),
            pybind11::arg("kb_id") = kb_id,
            pybind11::arg("display") = true);
        return result.cast<pybind11::list>()[0];
    }

    std::vector<pybind11::object> get_nodes(
        const std::string& group_name,
        const std::string& kb_id,
        const std::string& doc_id
    ) const {
        pybind11::gil_scoped_acquire gil;
        pybind11::object fn = _py_store.attr("get_nodes");
        pybind11::object result = fn(
            pybind11::arg("group_name") = group_name,
            pybind11::arg("kb_id") = kb_id,
            pybind11::arg("doc_ids") = std::vector<std::string>({doc_id}));
        return result.cast<std::vector<pybind11::object>>();
    }

private:
    // Keep the underlying Python store object alive for callback invocations.
    pybind11::object _py_store;
    std::unordered_map<std::string, NodeGroup> _node_groups_map;

    // Cache by Python object identity to ensure one wrapper per store instance.
    static std::unordered_map<PyObject *, std::weak_ptr<DocumentStore>> &store_cache() {
        static std::unordered_map<PyObject *, std::weak_ptr<DocumentStore>> cache;
        return cache;
    }
};

} // namespace lazyllm
