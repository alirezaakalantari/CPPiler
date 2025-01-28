Project Overview: C++ Compiler

This project is an implementation of a C++ compiler designed for educational purposes. It includes a lexical analyzer, predictive parser, token table, parse table, and additional features like syntax error handling and identifier definition tracking. The project is part of a data structures and algorithms course and demonstrates foundational concepts in compiler design, such as regular expressions, context-free grammar (CFG), and parsing techniques.

Features

Lexical Analyzer: Converts input C++ code into tokens using regular expressions.

Parser: Implements an LL(1) predictive parsing algorithm based on a predefined CFG.

Token and Parse Tables: Creates structured tables for tokens and grammar parsing.

Error Handling: Detects and reports syntax errors, such as missing semicolons or invalid assignments.

Search in Parse Tree: Locates the definition of identifiers within the parse tree.

Prerequisites

Understanding of DFA and CFG.

Familiarity with data structures such as arrays, stacks, linked lists, and trees.

Basic knowledge of hash functions and sorting algorithms.

How to Use

Input: Provide valid C++ code as input to the program.

Lexical Analysis: The lexer tokenizes the input code into categories such as keywords, identifiers, numbers, strings, and symbols.

Parsing: The parser checks the syntax based on the CFG and constructs a parse tree.

Token Table: View organized tokens with their respective hash values.

Error Handling: Any syntax errors encountered are displayed with line numbers and context.

Identifier Search: Search for variable definitions and initializations in the parse tree.

Example Input

#include <iostream>
using namespace std;
int main() {
    int x;
    int s = 0, t = 10;
    while (t >= 0) {
        cin >> x;
        t = t - 1;
        s = s + x;
    }
    cout << "sum=" << s;
    return 0;
}

Project Structure

main.py: Main Python script containing all functionality, including lexical analysis, parsing, and error handling.


پروژه: کامپایلر C++

این پروژه یک کامپایلر C++ را برای هدف‌های آموزشی پیاده‌سازی می‌کند. این پروژه شامل یک تحلیل‌گر لکسیکال، یک پارسر پیش‌بینی، جدول توکن‌ها، جدول پارس و ویژگی‌های اضافه مانند مدیریت خطاهای نحوی و ردیابی تعریف‌های متغیرها است.

ویژگی‌ها

تحلیل‌گر لکسیکال: تبدیل کد ورودی C++ به توکن‌ها با استفاده از عبارات منظم.

پارسر: پیاده‌سازی الگوریتم پارس LL(1) بر اساس CFG.

جدول توکن و پارس: ایجاد جدول‌های ساختاریافته برای توکن‌ها و تحلیل نحوی.

مدیریت خطا: شناسایی و گزارش خطاهای نحوی مانند فاقد سمی‌‌‌‌‌کولن در پایان دستورات.

جستجو در درخت پارس: یافتن تعریف متغیرها در درخت پارس.

ورودی نمونه


