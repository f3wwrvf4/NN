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




  Matrix a(100, 13);
  Matrix b(48, 100);
  Matrix c(b.row(), a.col());
  const float alpha = 0.7f;
  const float beta = 1.3f;

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

  NN::HelperEnable(true);
  {
    DWORD tick = GetTickCount();
    Matrix::Gemm(alpha, a, b, beta, c, c);
    float sum = MatrixSum(c);
    DWORD count = (GetTickCount() - tick);
    std::cout << " GPU:" << sum << " Tick: " << count << std::endl;
  }
  NN::HelperEnable(false);
  {
    DWORD tick = GetTickCount();
    Matrix::Gemm(alpha, a, b, beta, c, c);
    DWORD count = (GetTickCount() - tick);
    float sum = MatrixSum(c);
    std::cout << " CPU:" << sum << " Tick: " << count << std::endl;
  }

  NN::MathTerm();

  getchar();
}


