#include "doc_node.h"

namespace lazyllm {

DocNode::DocNode(const std::string& text) : _text(text) {}

void DocNode::set_text(const std::string& text) {
    _text = text;
}

const std::string& DocNode::get_text() const {
    return _text;
}

} // namespace lazyllm
