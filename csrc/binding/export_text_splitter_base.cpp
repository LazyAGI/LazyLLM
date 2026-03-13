#include "lazyllm.hpp"

#include "text_splitter_base.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

namespace {

using SplitFn = lazyllm::TextSplitterBase::SplitFn;

class PyTextSplitterBase final : public lazyllm::TextSplitterBase {
public:
    using lazyllm::TextSplitterBase::TextSplitterBase;

    std::vector<lazyllm::PDocNode> transform(const lazyllm::PDocNode node) const override {
        PYBIND11_OVERRIDE(
            std::vector<lazyllm::PDocNode>,
            lazyllm::TextSplitterBase,
            transform,
            node
        );
    }
};

py::object GetBaseModule() {
    return py::module_::import("lazyllm.tools.rag.transform.base");
}

py::object GetUnset() {
    return GetBaseModule().attr("_UNSET");
}

bool IsUnset(const py::object& value) {
    return value.is(GetUnset());
}

py::dict GetDefaultParams(const py::object& cls) {
    if (!py::hasattr(cls, "_default_params")) {
        cls.attr("_default_params") = py::dict();
    }
    return cls.attr("_default_params").cast<py::dict>();
}

unsigned ResolveUnsigned(
    const py::object& cls,
    const std::string& param_name,
    const py::object& value,
    unsigned default_value
) {
    if (value.is_none()) return default_value;
    if (!IsUnset(value)) {
        if (py::isinstance<py::int_>(value)) return value.cast<unsigned>();
        throw py::type_error(param_name + " must be an int");
    }

    py::dict defaults = GetDefaultParams(cls);
    if (defaults.contains(param_name.c_str())) {
        py::object v = py::reinterpret_borrow<py::object>(defaults[param_name.c_str()]);
        if (py::isinstance<py::int_>(v)) return v.cast<unsigned>();
    }
    return default_value;
}

py::object GetSplitClass() {
    return GetBaseModule().attr("_Split");
}

py::object MakeSplit(const std::string& text, bool is_sentence, int token_size) {
    return GetSplitClass()(py::arg("text") = text,
                           py::arg("is_sentence") = is_sentence,
                           py::arg("token_size") = token_size);
}

std::pair<std::vector<std::string>, bool> GetSplitsByFns(const std::string& text) {
    py::object base = GetBaseModule();
    py::object split_keep = base.attr("split_text_keep_separator");
    py::object nltk = base.attr("nltk");
    py::object tokenizer = nltk.attr("tokenize").attr("PunktSentenceTokenizer")();

    py::object splits_obj = split_keep(text, "\n\n\n");
    auto splits = splits_obj.cast<std::vector<std::string>>();
    if (splits.size() > 1) return {splits, true};

    splits_obj = tokenizer.attr("tokenize")(text);
    splits = splits_obj.cast<std::vector<std::string>>();
    if (splits.size() > 1) return {splits, true};

    py::object re = base.attr("re");
    splits_obj = re.attr("findall")(py::str(R"([^,.;。？！]+[,.;。？！]?)"), text);
    splits = splits_obj.cast<std::vector<std::string>>();
    if (splits.size() > 1) return {splits, false};

    splits_obj = split_keep(text, " ");
    splits = splits_obj.cast<std::vector<std::string>>();
    if (splits.size() > 1) return {splits, false};

    splits_obj = py::list(py::str(text));
    splits = splits_obj.cast<std::vector<std::string>>();
    return {splits, false};
}

void InitTiktokenTokenizer(
    const py::object& py_self,
    const std::string& encoding_name,
    const py::object& model_name,
    py::object allowed_special,
    py::object disallowed_special
);

void EnsureTokenCodec(const py::object& self) {
    py::dict obj_dict = py::getattr(self, "__dict__", py::dict()).cast<py::dict>();
    const bool has_encoder =
        obj_dict.contains("token_encoder") &&
        !py::reinterpret_borrow<py::object>(obj_dict["token_encoder"]).is_none();
    const bool has_decoder =
        obj_dict.contains("token_decoder") &&
        !py::reinterpret_borrow<py::object>(obj_dict["token_decoder"]).is_none();
    if (!has_encoder || !has_decoder) {
        InitTiktokenTokenizer(self, "gpt2", py::none(), py::none(), py::str("all"));
    }
}

int TokenSize(const py::object& self, const std::string& text) {
    EnsureTokenCodec(self);
    py::object encoder = self.attr("token_encoder");
    py::object tokens = encoder(text);
    return static_cast<int>(py::len(tokens));
}

void ExtendList(py::list& target, const py::list& src) {
    for (auto item : src) target.append(item);
}

py::list SplitRecursive(const py::object& self, const std::string& text, int chunk_size) {
    const int token_size = TokenSize(self, text);
    if (token_size <= chunk_size) {
        py::list out;
        out.append(MakeSplit(text, true, token_size));
        return out;
    }

    auto [text_splits, is_sentence] = GetSplitsByFns(text);
    py::list results;
    for (const auto& segment : text_splits) {
        const int seg_token_size = TokenSize(self, segment);
        if (seg_token_size <= chunk_size) {
            results.append(MakeSplit(segment, is_sentence, seg_token_size));
        } else {
            py::list sub = SplitRecursive(self, segment, chunk_size);
            ExtendList(results, sub);
        }
    }
    return results;
}

void InitTiktokenTokenizer(
    const py::object& py_self,
    const std::string& encoding_name,
    const py::object& model_name,
    py::object allowed_special,
    py::object disallowed_special
) {
    (void)allowed_special;
    (void)disallowed_special;

    std::string tokenizer_name = encoding_name;
    if (!model_name.is_none()) {
        if (!py::isinstance<py::str>(model_name)) {
            throw py::type_error("model_name must be a string");
        }
        tokenizer_name = model_name.cast<std::string>();
    }
    auto tokenizer = std::make_shared<TiktokenTokenizer>(tokenizer_name);

    py::object encode = py::cpp_function([tokenizer](const std::string& text) {
        return tokenizer->encode(text);
    });
    py::object decode = py::cpp_function([tokenizer](const std::vector<int>& ids) {
        py::bytes raw(tokenizer->decode(ids));
        return raw.attr("decode")("utf-8", "replace");
    });

    py_self.attr("token_encoder") = encode;
    py_self.attr("token_decoder") = decode;
}

void InitTokenizerFromObject(const py::object& py_self, const py::object& tokenizer, bool huggingface) {
    py::object encode;
    py::object decode;
    if (huggingface) {
        encode = py::cpp_function([tokenizer](const std::string& text) {
            return tokenizer.attr("encode")(text, py::arg("add_special_tokens") = false);
        });
        decode = py::cpp_function([tokenizer](const std::vector<int>& ids) {
            return tokenizer.attr("decode")(ids, py::arg("skip_special_tokens") = true);
        });
    } else {
        encode = py::cpp_function([tokenizer](const std::string& text) {
            return tokenizer.attr("encode")(text);
        });
        decode = py::cpp_function([tokenizer](const std::vector<int>& ids) {
            return tokenizer.attr("decode")(ids);
        });
    }

    py_self.attr("token_encoder") = encode;
    py_self.attr("token_decoder") = decode;
}

} // namespace

void exportTextSpliterBase(py::module& m) {
    py::class_<lazyllm::TextSplitterBase, lazyllm::NodeTransform, PyTextSplitterBase>(
        m, "_TextSplitterBase", py::dynamic_attr())
        .def(py::init([](py::object chunk_size,
                         py::object overlap,
                         py::object num_workers) {
                py::object cls = py::module_::import("lazyllm.lazyllm_cpp").attr("_TextSplitterBase");

                unsigned cs = ResolveUnsigned(cls, "chunk_size", chunk_size, 1024);
                unsigned ov = ResolveUnsigned(cls, "overlap", overlap, 200);
                unsigned nw = ResolveUnsigned(cls, "num_workers", num_workers, 0);

                if (ov > cs) {
                    throw py::value_error(
                        "Got a larger chunk overlap (" + std::to_string(ov) + ") than chunk size (" +
                        std::to_string(cs) + "), should be smaller.");
                }
                if (cs == 0) {
                    throw py::value_error("chunk size should > 0 and overlap should >= 0");
                }

                return std::make_unique<PyTextSplitterBase>(cs, ov, nw, "gpt2");
            }),
            py::arg("chunk_size") = py::none(),
            py::arg("overlap") = py::none(),
            py::arg("num_workers") = py::none()
        )
        .def_property_readonly("_chunk_size", &lazyllm::TextSplitterBase::chunk_size)
        .def_property_readonly("_overlap", &lazyllm::TextSplitterBase::overlap)
        .def("__getattr__",
            [](py::object self, const std::string& name) -> py::object {
                if (name == "token_encoder" || name == "token_decoder") {
                    EnsureTokenCodec(self);
                    py::dict obj_dict = py::getattr(self, "__dict__", py::dict()).cast<py::dict>();
                    if (obj_dict.contains(name.c_str())) {
                        return py::reinterpret_borrow<py::object>(obj_dict[name.c_str()]);
                    }
                }
                throw py::attribute_error(
                    "'" + std::string(py::str(py::type::of(self).attr("__name__"))) +
                    "' object has no attribute '" + name + "'");
            },
            py::arg("name")
        )
        .def("split_text",
            [](lazyllm::TextSplitterBase& self, const std::string& text, int metadata_size) {
                if (text.empty()) return std::vector<std::string>{""};

                const int chunk_size = self.chunk_size();
                const int effective_chunk_size = chunk_size - metadata_size;
                if (effective_chunk_size <= 0) {
                    throw py::value_error(
                        "Metadata length (" + std::to_string(metadata_size) + ") is longer than chunk size (" +
                        std::to_string(chunk_size) + "). Consider increasing the chunk size or decreasing the size of "
                        "your metadata to avoid this.");
                } else if (effective_chunk_size < 50) {
                    try {
                        py::object log = py::module_::import("lazyllm").attr("LOG");
                        log.attr("warning")(
                            "Metadata length (" + std::to_string(metadata_size) + ") is close to chunk size (" +
                            std::to_string(chunk_size) + "). Resulting chunks are less than 50 tokens. "
                            "Consider increasing the chunk size or decreasing the size of your metadata to avoid this.");
                    } catch (const py::error_already_set&) {
                    }
                }

                py::object py_self = py::cast(&self, py::return_value_policy::reference);
                py::object splits = py_self.attr("_split")(text, effective_chunk_size);
                py::object chunks = py_self.attr("_merge")(splits, effective_chunk_size);
                return chunks.cast<std::vector<std::string>>();
            },
            py::arg("text"), py::arg("metadata_size")
        )
        .def_static("split_text_keep_separator",
            [](const std::string& text, const std::string& separator) {
                auto views = lazyllm::TextSplitterBase::split_text_while_keeping_separator(text, separator);
                std::vector<std::string> out;
                out.reserve(views.size());
                for (auto view : views) out.emplace_back(view);
                return out;
            },
            py::arg("text"), py::arg("separator")
        )
        .def("from_tiktoken_encoder",
            [](py::object self,
               const std::string& encoding_name,
               py::object model_name,
               py::object allowed_special,
               py::object disallowed_special,
               py::kwargs /*kwargs*/) {
                InitTiktokenTokenizer(self, encoding_name, model_name, allowed_special, disallowed_special);
                return self;
            },
            py::arg("encoding_name") = "gpt2",
            py::arg("model_name") = py::none(),
            py::arg("allowed_special") = py::none(),
            py::arg("disallowed_special") = "all"
        )
        .def("from_tiktoken_encoding",
            [](py::object self, const std::string& encoding_name) {
                InitTiktokenTokenizer(self, encoding_name, py::none(), py::none(), py::str("all"));
                return self;
            },
            py::arg("encoding_name") = "gpt2"
        )
        .def("from_tokenizer",
            [](py::object self, py::object tokenizer) {
                InitTokenizerFromObject(self, tokenizer, false);
                return self;
            },
            py::arg("tokenizer")
        )
        .def("from_huggingface_tokenizer",
            [](py::object self, py::object tokenizer) {
                InitTokenizerFromObject(self, tokenizer, true);
                return self;
            },
            py::arg("tokenizer")
        )
        .def("set_split_fns",
            [](py::object /*self*/, const py::object& /*split_fns*/, py::object /*sub_split_fns*/) {
                return;
            },
            py::arg("split_fns"), py::arg("sub_split_fns") = py::none()
        )
        .def("add_split_fn",
            [](py::object /*self*/, const py::object& /*split_fn*/, py::object /*index*/) {
                return;
            },
            py::arg("split_fn"), py::arg("index") = py::none()
        )
        .def("clear_split_fns",
            [](py::object /*self*/) {
                return;
            }
        )
        .def("_token_size",
            [](py::object self, const std::string& text) {
                return TokenSize(self, text);
            },
            py::arg("text")
        )
        .def("_get_metadata_size",
            [](py::object self, py::object node) {
                py::object meta_mode = GetBaseModule().attr("MetadataMode");
                std::string embed = node.attr("get_metadata_str")(meta_mode.attr("EMBED")).cast<std::string>();
                std::string llm = node.attr("get_metadata_str")(meta_mode.attr("LLM")).cast<std::string>();
                return std::max(TokenSize(self, embed), TokenSize(self, llm));
            },
            py::arg("node")
        )
        .def("_get_splits_by_fns",
            [](py::object /*self*/, const std::string& text) {
                auto [splits, is_sentence] = GetSplitsByFns(text);
                return py::make_tuple(splits, is_sentence);
            },
            py::arg("text")
        )
        .def("_split",
            [](py::object self, const std::string& text, int chunk_size) {
                return SplitRecursive(self, text, chunk_size);
            },
            py::arg("text"), py::arg("chunk_size")
        )
        .def("_merge",
            [](lazyllm::TextSplitterBase& self, py::list splits, int chunk_size) {
                if (py::len(splits) == 0) return std::vector<std::string>{};
                if (py::len(splits) == 1) {
                    py::object s = splits[0];
                    return std::vector<std::string>{py::cast<std::string>(s.attr("text"))};
                }

                struct SplitData { std::string text; bool is_sentence; int token_size; };
                std::vector<SplitData> data;
                data.reserve(py::len(splits));
                for (auto item : splits) {
                    py::object s = py::reinterpret_borrow<py::object>(item);
                    data.push_back({
                        s.attr("text").cast<std::string>(),
                        s.attr("is_sentence").cast<bool>(),
                        s.attr("token_size").cast<int>()
                    });
                }

                const int overlap = self.overlap();
                py::object py_self = py::cast(&self, py::return_value_policy::reference);
                EnsureTokenCodec(py_self);
                py::object encoder = py_self.attr("token_encoder");
                py::object decoder = py_self.attr("token_decoder");

                if (data.back().token_size == chunk_size && overlap > 0) {
                    SplitData end_split = data.back();
                    data.pop_back();
                    auto tokens = encoder(end_split.text).cast<std::vector<int>>();
                    const size_t half = tokens.size() / 2;
                    auto mid = static_cast<std::vector<int>::difference_type>(half);
                    std::vector<int> prefix(tokens.begin(), tokens.begin() + mid);
                    std::vector<int> suffix(tokens.begin() + mid, tokens.end());
                    std::string p_text = decoder(prefix).cast<std::string>();
                    std::string n_text = decoder(suffix).cast<std::string>();
                    data.push_back({p_text, end_split.is_sentence, TokenSize(py_self, p_text)});
                    data.push_back({n_text, end_split.is_sentence, TokenSize(py_self, n_text)});
                }

                SplitData end_split = data.back();
                std::vector<std::string> result;
                for (int idx = static_cast<int>(data.size()) - 2; idx >= 0; --idx) {
                    SplitData start_split = data[static_cast<size_t>(idx)];
                    if (start_split.token_size <= overlap && end_split.token_size <= chunk_size - overlap) {
                        end_split.text = start_split.text + end_split.text;
                        end_split.is_sentence = start_split.is_sentence && end_split.is_sentence;
                        end_split.token_size += start_split.token_size;
                        continue;
                    }

                    if (end_split.token_size > chunk_size) {
                        throw py::value_error(
                            "split token size (" + std::to_string(end_split.token_size) + ") is greater than chunk size (" +
                            std::to_string(chunk_size) + ").");
                    }

                    const int remaining_space = chunk_size - end_split.token_size;
                    const int overlap_len = std::min({overlap, remaining_space, start_split.token_size});
                    if (overlap_len > 0) {
                        auto start_tokens = encoder(start_split.text).cast<std::vector<int>>();
                        std::vector<int> overlap_tokens(start_tokens.end() - overlap_len, start_tokens.end());
                        std::string overlap_text = decoder(overlap_tokens).cast<std::string>();
                        end_split.text = overlap_text + end_split.text;
                        end_split.token_size += overlap_len;
                    }

                    result.insert(result.begin(), end_split.text);
                    end_split = start_split;
                }

                result.insert(result.begin(), end_split.text);
                return result;
            },
            py::arg("splits"), py::arg("chunk_size")
        )
        .def("forward",
            [](py::object self, py::object node, py::kwargs /*kwargs*/) {
                const std::string text = node.attr("get_text")().cast<std::string>();
                const int metadata_size = self.attr("_get_metadata_size")(node).cast<int>();
                py::object chunks = self.attr("split_text")(text, metadata_size);

                std::vector<lazyllm::PDocNode> out;
                py::object doc_cls = py::module_::import("lazyllm.tools.rag.doc_node").attr("DocNode");
                for (auto item : chunks) {
                    py::object chunk = py::reinterpret_borrow<py::object>(item);
                    if (chunk.is_none()) continue;
                    if (py::isinstance(chunk, doc_cls)) {
                        out.push_back(chunk.cast<lazyllm::PDocNode>());
                        continue;
                    }
                    std::string chunk_text = chunk.cast<std::string>();
                    if (chunk_text.empty()) continue;
                    out.push_back(std::make_shared<lazyllm::DocNode>(std::move(chunk_text)));
                }
                return out;
            },
            py::arg("node")
        )
        .def("_get_param_value",
            [](py::object self, const std::string& param_name, py::object value, py::object default_value) {
                if (!IsUnset(value)) return value;
                py::object cls = py::type::of(self);
                py::dict defaults = GetDefaultParams(cls);
                if (defaults.contains(param_name.c_str())) {
                    return py::reinterpret_borrow<py::object>(defaults[param_name.c_str()]);
                }
                return default_value;
            },
            py::arg("param_name"), py::arg("value"), py::arg("default")
        );

    py::object py_cls = m.attr("_TextSplitterBase");
    py::object builtins = py::module_::import("builtins");

    py_cls.attr("_get_class_lock") = builtins.attr("classmethod")(
        py::cpp_function([](py::object klass) {
            if (!py::hasattr(klass, "_default_params_lock")) {
                klass.attr("_default_params_lock") = py::module_::import("threading").attr("RLock")();
            }
            return klass.attr("_default_params_lock");
        })
    );

    py_cls.attr("set_default") = builtins.attr("classmethod")(
        py::cpp_function([](py::object klass, py::kwargs kwargs) {
            py::dict defaults = GetDefaultParams(klass);
            for (auto item : kwargs) {
                defaults[item.first] = item.second;
            }
            klass.attr("_default_params") = defaults;
        })
    );

    py_cls.attr("get_default") = builtins.attr("classmethod")(
        py::cpp_function([](py::object klass, py::object param_name) -> py::object {
            py::dict defaults = GetDefaultParams(klass);
            if (param_name.is_none()) return py::object(defaults);
            const std::string key = param_name.cast<std::string>();
            if (defaults.contains(key.c_str())) {
                return py::reinterpret_borrow<py::object>(defaults[key.c_str()]);
            }
            return py::none();
        })
    );

    py_cls.attr("reset_default") = builtins.attr("classmethod")(
        py::cpp_function([](py::object klass) {
            klass.attr("_default_params") = py::dict();
        })
    );
}
