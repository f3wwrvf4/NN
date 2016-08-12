// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>

#include <NN_net.h>
#include <NN_math.h>
#include <NN_mnist.h>
#include <NN_iris.h>

float MatrixSum(const NN::Matrix& mat)
{
  float sum = 0;
  for (int i = 0; i < mat.row(); ++i) {
    for (int j = 0; j < mat.col(); ++j) {
      sum += mat(i, j);
    }
  }

  return sum;
}

int main()
{
  using namespace NN;

  NN::MathInit();

#if 1
  Matrix a(100, 13);
  Matrix b(48, 100);
#else
  Matrix a(2000, 3000);
  Matrix b(1000, 2000);
#endif
  Matrix c(b.row(), a.col());

#if 1
  std::srand(0xababcdcd);
  a.random();
  b.random();
  c.random();

#else
  int val = 0;
  for (int i = 0; i < a.row(); ++i) {
    for (int j = 0; j < a.col(); ++j) {
      a(i, j) = 0.5f + val++ * ((val%2)?-1.f:1.f);
    }
  }
  val = 0;
  for (int i = 0; i < b.row(); ++i) {
    for (int j = 0; j < b.col(); ++j) {
      b(i, j) = 0.5f + val++ * ((val%2)?1.f:-1.f);
    }
  }
  val = 0;
  for (int i = 0; i < c.row(); ++i) {
    for (int j = 0; j < c.col(); ++j) {
      c(i, j) = 0.0f;
    }
  }
#endif

  std::cout << MatrixSum(a) << " " << MatrixSum(b) << " " << MatrixSum(c) << "\n";

  DWORD tick = GetTickCount();
  Matrix::Gemm(1.0f, a, b, 1.33f, c, c);

//  std::cout << c << std::endl;
  std::cout << "Mul... " << (GetTickCount() - tick) << std::endl; // 125 / 24312

  std::cout << MatrixSum(c) << std::endl;  // 11.2602 / -3295.46

  NN::MathTerm();

  getchar();
}


