# Contributing to Tribol

We welcome contributions to Tribol. To do so please submit a pull request through our
Tribol github page at https://github.com/LLNL/Tribol.

All contributions to Tribol must be made under the MIT License.

Any questions can be sent to wopschall1@llnl.gov.

## C++ implementation namespaces

Use `tribol::detail` for shared implementation helpers that must be declared in headers but are not supported public
interfaces. Prefer a private class member when a helper belongs to one class, and use an unnamed namespace when a helper
is confined to one `.cpp` file. Do not place user-facing types or functions in `tribol::detail`.

# Attribution

The Tribol project uses git's commit history to track contributions from individual developers. 
