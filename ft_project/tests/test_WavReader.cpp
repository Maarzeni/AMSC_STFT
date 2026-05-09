#include <gtest/gtest.h>
// #include "audio/WavReader.hpp" <-- Lo scommenterai quando esisterà

// Test di prova: Verifica che 2+2 faccia 4 per assicurarsi che GTest funzioni
TEST(InfrastructureTest, GTestIsWorking) {
    int expected = 4;
    int actual = 2 + 2;
    EXPECT_EQ(expected, actual);
}