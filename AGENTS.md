## Contributor guidance

- Follow the project conventions in `CONTRIBUTING.md`, including its guidance for C++ implementation namespaces.

## Formatting

- Tribol uses a 120-character column width. Wrap prose comments and documentation near 120 columns rather than the
  common 80-column convention.
- Always enclose `if` and `for` statement bodies in braces, including single-statement bodies.
- Run `clang-format-19 -i` on every C or C++ source and header file changed before finalizing the work.

## Naming

- Use `ct` as shorthand for `count` in new identifiers, following the surrounding capitalization: `pair_ct`,
  `ValueCt`, and `pairCt()`. Keep existing shared and external API names.

## Numeric literals

- Avoid unexplained numeric literals. Name values by their purpose, calculate counts from existing data where practical,
  and explain choices that affect behavior. Keep obvious indices, mathematical constants, and readable coordinate tables
  inline.

## Documentation

- Document new headers, types, functions, and members with Doxygen; cover all parameters using `@param` and `@tparam`.
