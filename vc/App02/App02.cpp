// App02.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
#include <string.h>

template <int COL, int ROW>
struct Matrix
{
  float buf[ROW][COL];

  template <int ROW2>
  void mul(const Matrix<ROW, ROW2>& m2, Matrix<COL, ROW2>& out)
  {
    memset(out.buf, 0, sizeof(out.buf));

    for (int i = 0; i < COL; ++i) {
      for (int j = 0; j < ROW2; ++j) {
        for (int k = 0; k < ROW; ++k) {
          out.buf[i][j] += this->buf[i][k] * m2.buf[k][j];
        }
      }
    }
  }
};

template<int T1 = 0, int T2 = 0, int T3 = 0, int T4 = 0, int T5 = 0 >
struct IntList
{
  typedef IntList<T2, T3, T4, T5> Tail;
  enum 
  {
    Length = Tail::Length + 1
  };

  int length() { return Length; }
  int size() { return T1; }

  int val[T1];
  Tail tail;
};

template<>
struct IntList<0>
{
  enum { Length = 0 };
};

const int BatchNum = 10;

template< int T1 = 0, int T2 = 0, int T3 = 0, int T4 = 0, int T5 = 0 >
struct Layer
{
  typedef Layer<T2, T3, T4, T5> Tail;
  Tail tail;

  float node_value[T1 + Tail::IsMiddle];

  Matrix<Size, Tail::Size> weight;
  Matrix<BatchNum, 

  enum
  {
    IsMiddle = 1,

    Size = T1 + Tail::IsMiddle,
  };
};

template<>
struct Layer<0>
{
  enum
  {
    IsMiddle = 0,
  };
};

int main()
{
  Layer<2, 3, 4> l;

  return 0;
}

