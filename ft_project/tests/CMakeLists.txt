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

# Recursive FFT Test
add_executable(test_RecursiveFFT test_RecursiveFFT.cpp)
target_link_libraries(test_RecursiveFFT PRIVATE amsc_stft_core GTest::gtest_main)
target_compile_definitions(test_RecursiveFFT PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME RecursiveFFTTest COMMAND test_RecursiveFFT)

# Iterative FFT Test 
add_executable(test_IterativeFFT test_IterativeFFT.cpp)
target_link_libraries(test_IterativeFFT PRIVATE amsc_stft_core GTest::gtest_main)
add_test(NAME IterativeFFTTest COMMAND test_IterativeFFT)

# --- Parallel FFT Test ---
add_executable(test_ParallelFFT test_ParallelFFT.cpp)
target_link_libraries(test_ParallelFFT PRIVATE amsc_stft_core GTest::gtest_main)
add_test(NAME ParallelFFTTest COMMAND test_ParallelFFT)

# --- Parallel FFT Test with OpenMP ---
add_executable(test_ParallelFFT_OpenMP test_ParallelFFT_OpenMP.cpp)
target_link_libraries(test_ParallelFFT_OpenMP PRIVATE amsc_stft_core GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ParallelFFTOpenMPTest COMMAND test_ParallelFFT_OpenMP)