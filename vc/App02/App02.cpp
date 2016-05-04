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

template<  
  int T1 = 0, int T2 = 0, int T3 = 0, int T4 = 0, int T5 = 0
>
struct IntList
{
private:
  typedef IntList<T2, T3, T4, T5> Tail;
  enum 
  {
    Length = Tail::Length + 1
  };
public:

  int length() { return Length; }
  int size() { return T1; }

public:
  int val[T1];
private:
  Tail tail;
};

template<>
struct IntList<0>
{
  enum { Length = 0 };
};


struct Network
{
  struct InitParam
  {

  };

  Network()
  {

  }

  T inputs[L1]

};


int main()
{
  IntList<2, 3, 4> il;

  il.val[0] = 1;

  return 0;
}

