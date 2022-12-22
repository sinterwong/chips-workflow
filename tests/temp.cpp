/* Proj: cpp-functional-programming
 * File: for-excepted.cpp
 * Created Date: 2022/11/12
 * Author: YangYangYang (yangyangyang0110@outlook.com)
 * Description:
 * -----
 * Last Modified: 2022/11/12 19:17:17
 * Modified By: YangYangYang (yangyangyang0110@outlook.com)
 * -----
 * Copyright (c) 2022  . All rights reserved.
 */

#include <future>
#include <iostream>
#include <optional>
#include <variant>

// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4015.pdf

struct DivideByZero : public std::exception {
};

double safe_divide(double x, double y) noexcept(false)
{
    if (y == 0) throw DivideByZero();
    return x / y;
}

// void test() {
//   try {
//     safe_divide(20, 10);
//     safe_divide(10, 0);
//   } catch (const std::exception &e) { std::cerr << "reason: " << e.what() << std::endl;
//   }
// }

// template <typename T>
// concept MoveConstructorAble = requires(T x) {
//                                   {
//                                       T y = T(std::move(x));
//                                   } noexcept;
//                               };

/**
 * @brief 如何约束T的能力集呢?
 * @tparam T
 * @tparam E
 */
template <class T, class E> struct Excepted {
    template <typename... Args>
    constexpr static inline Excepted<T, E> success(Args &&...args)
    {
        Excepted result;
        result.has_value_ = true;
        new (&result.value_) T(std::forward<Args>(args)...);
        return result;
    }

    template <typename... Args>
    constexpr static inline Excepted<T, E> failure(Args &&...args)
    {
        Excepted result;
        result.has_value_ = false;
        new (&result.error_) E(std::forward<Args>(args)...);
        return result;
    }

public:
    constexpr T &value() noexcept(false)
    {
        if (!has_value_) throw std::logic_error("No value");
        return value_;
    }

    constexpr E &error() noexcept(false)
    {
        if (has_value_) throw std::logic_error("No error");
        return error_;
    }

    operator bool() const noexcept { return has_value_; }

    operator std::optional<T>() const noexcept
    {
        return has_value_ ? std::make_optional<T>(value_) : std::nullopt;
    }

public:
    Excepted()
        : has_value_(false){

          };

    Excepted(const Excepted &other)
        : has_value_(other.has_value_)
    {
        if (other.has_value_) new (&value_) T(other.value_);
        else new (&error_) E(other.error_);
    }

    Excepted(Excepted &&other) noexcept
        : has_value_(other.has_value_)
    {
        if (other.has_value_) new (&value_) T(std::move(other.value_));
        else new (&error_) E(std::move(other.error_));
    }

    // 使用副本去做swap, 保证数据一致性.
    Excepted &operator=(Excepted other)
    {
        swap(other);
        return *this;
    }

    ~Excepted() noexcept
    {
        if (has_value_) {
            value_.~T();
        } else {
            error_.~E();
        }
    };

private:
    // copy and swap
    void swap(Excepted &other)
    {
        using std::swap;
        if (has_value_) {
            if (other.has_value_) {
                swap(value_, other.value_);
            } else {
                // copy error.
                auto error_clone = std::move(other.error_);
                other.error_.~E();

                new (&other.value_) T(std::move(value_));
                value_.~T();

                new (&error_) E(std::move(error_clone));

                swap(has_value_, other.has_value_);
            }
        } else {
            if (other.has_value_) {
                other.swap(*this); // 递归调用.
            } else {
                swap(other.error_, other.error_);
            }
        }
    }

private:
    bool has_value_;
    union {
        T value_;
        E error_;
    };
};

struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(NonCopyable const &) = delete;
    NonCopyable &operator=(NonCopyable const &) = delete;
};

enum class ErrorCode { InvalidParam = -1 };

struct Foo {
    using excepted_type = Excepted<Foo, ErrorCode>;
    static inline excepted_type try_from(bool b) noexcept
    {
        return b ? excepted_type::success(Foo())
                 : excepted_type::failure(ErrorCode::InvalidParam);
    }

    friend std::ostream &operator<<(std::ostream &os, Foo const &foo) noexcept
    {
        return os << foo.data << std::endl;
    }

public:
    Foo() = default;
    Foo(Foo &&) noexcept = default;
    ~Foo() noexcept = default;

private:
    std::string data = "I'm a Foo";
};

template <typename F, typename Ret = std::result_of_t<F()>,
          typename Except = Excepted<Ret, std::exception_ptr>>
Except m_try(const F &f) noexcept
{
    try {
        return Except::success(f());
    } catch (...) {
        return Except::failure(std::current_exception());
    }
}

template <typename T, typename Variant, typename Except = Excepted<T, std::exception_ptr>>
Except m_get_if(const Variant &variant) noexcept
{
    auto *p = std::get_if<T>(variant);
    return p ? Except::success(p)
             : Except::failure("Variant not found the desired type.");
}

template <typename T, typename E> T &get_or_throw(Excepted<T, E> &except) noexcept(false)
{
    if (except) {
        except.value();
    } else {
        std::rethrow_exception(except.error());
    }
}

void test()
{
    std::cout << Foo::try_from(true).value()
              << static_cast<int>(Foo::try_from(false).error()) << std::endl;
}

int main()
{
    test();
    return 0;
}