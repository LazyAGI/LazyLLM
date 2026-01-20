#pragma once

#include <string>

namespace lazyllm {

class DocNode {
public:
    DocNode() = default;
    explicit DocNode(const std::string& text);

    void set_text(const std::string& text);
    const std::string& get_text() const;

private:
    std::string _text;
};

} // namespace lazyllm
