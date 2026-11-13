// Reproduce with:
// clang++ -std=c++20 -O0 -g -fplugin=/path/to/ClangEnzyme-19.so scripts/enzyme_o0_reproducer.cpp
//
// Enzyme 0.0.266 and 0.0.309 fail to identify the second double in the aggregate returned by operator-().
// Adding -Xclang -disable-O0-optnone fixes this reduced case, but Tribol's fixed-rule derivatives still require the
// optimized IR produced for EnergyMortar.cpp.

extern int enzyme_dup;

template <typename Return, typename... Args>
Return __enzyme_autodiff( Args... );

template <int Size>
struct Array {
  double values[Size];

  Array& operator-=( const Array& other )
  {
    for ( int i = 0; i < Size; ++i ) {
      values[i] -= other.values[i];
    }
    return *this;
  }
};

template <int Size>
Array<Size> operator-( const Array<Size>& first, const Array<Size>& second )
{
  Array<Size> result( first );
  result -= second;
  return result;
}

template <int Size>
struct Vector {
  Array<Size> data;

  Vector( const Array<Size>& first, const Array<Size>& second ) : data( second - first ) {}

  double squaredNorm() const
  {
    double result{};
    for ( int i = 0; i < Size; ++i ) {
      result += data.values[i] * data.values[i];
    }
    return result;
  }
};

__attribute__( ( noinline ) ) double squaredDistance( const double* values )
{
  const Array<2> first{ values[0], values[1] };
  const Array<2> second{ values[2], values[3] };
  return Vector<2>( first, second ).squaredNorm();
}

int main()
{
  double values[4]{ 1.0, 2.0, 4.0, 6.0 };
  double derivatives[4]{};
  __enzyme_autodiff<void>( (void*)squaredDistance, enzyme_dup, values, derivatives );
  return derivatives[0] == -6.0 && derivatives[1] == -8.0 && derivatives[2] == 6.0 && derivatives[3] == 8.0 ? 0 : 1;
}
