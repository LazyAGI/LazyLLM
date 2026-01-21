#include <gtest/gtest.h>

#include "doc_node.h"

TEST(DocNode, DefaultEmpty) {
    lazyllm::DocNode node;
    EXPECT_EQ(node.get_text(), "");
}

TEST(DocNode, SetGet) {
    lazyllm::DocNode node("hello");
    EXPECT_EQ(node.get_text(), "hello");

    node.set_text("world");
    EXPECT_EQ(node.get_text(), "world");
}
