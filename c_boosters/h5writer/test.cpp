#include <gtest/gtest.h>

extern "C" {
#include "main.h"
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
// Expect two strings not to be equal.
EXPECT_STRNE("hello", "world");
// Expect equality.
EXPECT_EQ(7 * 6, 42);
}

TEST(HelloTest2, BasicAssertions) {
    PyObject *argList = Py_BuildValue("si", "hello", 42);
    int res = write_hdf5_metadata(argList, argList);
    ASSERT_EQ(res,0);
}