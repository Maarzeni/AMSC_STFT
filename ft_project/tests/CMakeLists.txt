# Automatically download Google Test from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0 # Use the latest stable release
)
# For Windows: force the use of static libraries for GTest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- DEFINE YOUR TESTS HERE ---

# WavReader Test
# Create an executable named "test_WavReader" using the "test_WavReader.cpp" file
add_executable(test_WavReader test_WavReader.cpp)
# Link the test to the main library (amsc_stft_core) and Google Test
target_link_libraries(test_WavReader PRIVATE amsc_stft_core GTest::gtest_main)
# Register the test with the CMake testing system
add_test(NAME WavReaderTest COMMAND test_WavReader)

# --- C++20 Concepts Test ---
# Create an executable to test compile-time concept validation
add_executable(test_Concept test_Concept.cpp)
target_link_libraries(test_Concept PRIVATE amsc_stft_core GTest::gtest_main)
add_test(NAME ConceptTest COMMAND test_Concept)


# Example 2: Recursive FFT Test (left here for future implementation)
# add_executable(test_RecursiveFFT test_RecursiveFFT.cpp)
# target_link_libraries(test_RecursiveFFT PRIVATE amsc_stft_core GTest::gtest_main)
# add_test(NAME RecursiveFFTTest COMMAND test_RecursiveFFT)