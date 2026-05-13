/**
 * @file WindowConcept.hpp
 * @brief C++20 Concept definition for window function types.
 *
 * @details
 * This file defines a **C++20 Concept** called `WindowFunction`.
 * A Concept is a compile-time predicate: a named set of requirements
 * that a type must satisfy to be usable in a given template.
 *
 * ─── What is a Concept? ───────────────────────────────────────────────────────
 * Before C++20, if you wrote a template and passed the wrong type, you would
 * get cryptic error messages deep inside the template instantiation.
 *
 * With Concepts, the error is caught at the call site with a clear message:
 * @code
 *   // Without concept:
 *   STFTAnalyzer<FFT, int> bad;
 *   // Error: no member named 'apply' in 'int'
 *   //   in BaseWindow.hpp:147:23 ... (20 lines of template noise)
 *
 *   // With concept:
 *   STFTAnalyzer<FFT, int> bad;
 *   // Error: 'int' does not satisfy 'WindowFunction'
 *   //   note: 'int' does not have method 'apply(std::vector<double>&)'
 * @endcode
 *
 * ─── Syntax breakdown ─────────────────────────────────────────────────────────
 * @code
 *   template<typename W>
 *   concept WindowFunction = requires(...) { ... };
 * @endcode
 *
 * - `template<typename W>` : W is the type being checked.
 * - `concept WindowFunction` : we are defining a concept with this name.
 * - `= requires(...) { ... }` : a **requires-expression** that lists what W
 *   must provide. Each line inside the braces is a **requirement**.
 *
 * ─── Types of requirements ────────────────────────────────────────────────────
 * Inside a requires-expression you can write:
 *
 *   1. **Simple expression requirement**: `{ expr }` — the expression must
 *      be valid (compilable).
 *
 *   2. **Type requirement**: `{ expr } -> std::same_as<T>` — the expression
 *      must be valid AND return a type that matches T (or is convertible to T).
 *
 *   3. **Nested requirement**: `requires condition;` — a boolean constraint.
 *
 * ─── This concept is used by STFTAnalyzer ─────────────────────────────────────
 * @code
 *   template<typename FFT, WindowFunction Window>
 *   class STFTAnalyzer { ... };
 * @endcode
 * The word `WindowFunction` replaces `typename` — the compiler will check
 * the concept before instantiating the template.
 *
 * @note This file does NOT include BaseWindow.hpp or IWindow.hpp intentionally.
 *       A Concept describes a structural requirement ("has these methods"),
 *       not an inheritance requirement ("is derived from IWindow").
 *       This means you could satisfy WindowFunction with a completely
 *       independent class that never heard of IWindow — duck typing.
 *
 * @author  AMSC_STFT Project
 * @version 1.0
 */

#pragma once

#include <cstddef>   // std::size_t
#include <vector>    // std::vector
#include <concepts>  // std::same_as, std::constructible_from

namespace stft {

/**
 * @concept WindowFunction
 * @brief Constrains a type W to be a valid window function for STFT use.
 *
 * @details
 * A type W satisfies `WindowFunction` if and only if:
 *
 * | Requirement                   | Why it's needed                          |
 * |-------------------------------|------------------------------------------|
 * | W(std::size_t)                | Must be constructible with a frame size  |
 * | w.size() → std::size_t        | STFTAnalyzer needs to know the frame size |
 * | w.apply(vector<double>&)      | Core operation: window a signal frame    |
 * | w.coefficients() →            | Needed for normalization / export        |
 * |   const vector<double>&       |                                          |
 * | w.coherentGain() → double     | Needed for magnitude normalization       |
 * | w.powerBandwidth() → double   | Needed for energy analysis               |
 *
 * @tparam W  The type to check against this concept.
 *
 * @par Example usage in STFTAnalyzer:
 * @code
 * template<typename FFT, stft::WindowFunction Window>
 * class STFTAnalyzer {
 *     // Window is guaranteed to have apply(), size(), etc.
 * };
 * @endcode
 *
 * @par Example: verifying your own class satisfies the concept:
 * @code
 * static_assert(stft::WindowFunction<HannWindow>,
 *               "HannWindow must satisfy WindowFunction!");
 * @endcode
 */
template<typename W>
concept WindowFunction =

    // ── Requirement 1: constructible from a size ─────────────────────────────
    // `std::constructible_from<W, std::size_t>` is a standard library concept
    // that checks: can we write `W obj(someSize_t_value)`?
    std::constructible_from<W, std::size_t>

    // The `&&` here is not the "address-of" operator — it is the logical AND
    // that chains multiple requirements in a concept definition.
    &&

    // ── Requirements 2–6: method signatures ──────────────────────────────────
    // `requires(...)` introduces a requires-expression.
    // We declare local variables to use in the checks — they are never
    // actually created at runtime; they exist only for the type checker.
    requires(
        W w,                          // a mutable instance of W
        const W cw,                   // a const instance of W
        std::vector<double>& sig      // a reference to a signal vector
    )
    {
        // Each line checks that a specific expression is valid.
        // The `-> std::same_as<T>` part checks the exact return type.

        /**
         * w.size() must return exactly std::size_t (not int, not long).
         * `std::same_as<T>` is a standard concept that checks type identity.
         */
        { cw.size() } -> std::same_as<std::size_t>;

        /**
         * w.apply(signal) must accept a non-const vector reference and return void.
         * No return type check needed for void — we just check it compiles.
         */
        { w.apply(sig) } -> std::same_as<void>;

        /**
         * w.coefficients() must return a const reference to a double vector.
         * `std::same_as<const std::vector<double>&>` verifies the exact type.
         */
        { cw.coefficients() } -> std::same_as<const std::vector<double>&>;

        /**
         * w.coherentGain() must return a double.
         */
        { cw.coherentGain() } -> std::same_as<double>;

        /**
         * w.powerBandwidth() must return a double.
         */
        { cw.powerBandwidth() } -> std::same_as<double>;
    };

// ─── Convenience alias ────────────────────────────────────────────────────────
/**
 * @brief Helper to produce a clear static_assert message.
 *
 * @details
 * You can use this macro at the bottom of each concrete window header to
 * verify at compile time that the class correctly satisfies the concept:
 *
 * @code
 * ASSERT_WINDOW_FUNCTION(HannWindow);
 * // Equivalent to:
 * // static_assert(stft::WindowFunction<HannWindow>, "...");
 * @endcode
 *
 * Note: macros (`#define`) are a preprocessor feature — the compiler
 * textually substitutes the macro before compiling. They have no type
 * safety, so we use them sparingly and only for diagnostic helpers.
 */
#define ASSERT_WINDOW_FUNCTION(Type)                                   \
    static_assert(                                                     \
        ::stft::WindowFunction<Type>,                                  \
        #Type " does not satisfy the WindowFunction concept. "           \
        "Check that it has: size(), apply(), coefficients(), "           \
        "coherentGain(), powerBandwidth()."                              \
    )

} // namespace stft