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

void MatrixFill(NN::Matrix& mat, float val)
{
  for (int i = 0; i < mat.row(); ++i) {
    for (int j = 0; j < mat.col(); ++j) {
      mat(i, j) = val;
    }
  }
}


int main()
{
  using namespace NN;

  NN::MathInit();

  Matrix a(rand()%100+10, rand()%100+10);
  Matrix b(rand()%100+10, a.row());
  Matrix cpu_out(b.row(), a.col());
  Matrix gpu_out;
  const float alpha = 1.0f;
  const float beta = 1.0f;

  a.random();
  b.random();
  cpu_out.random();
  gpu_out = cpu_out;

  std::cout << MatrixSum(a) << " " << MatrixSum(b) << "\n"
    << MatrixSum(cpu_out) << MatrixSum(gpu_out) << "\n";

  NN::HelperEnable(true);
  {
    DWORD tick = GetTickCount();
    Matrix::Gemm(alpha, a, b, beta, cpu_out, cpu_out);
    float sum = MatrixSum(cpu_out);
    DWORD count = (GetTickCount() - tick);
    std::cout << " GPU:" << sum << " Tick: " << count << std::endl;
  }
  NN::HelperEnable(false);
  {
    DWORD tick = GetTickCount();
    Matrix::Gemm(alpha, a, b, beta, gpu_out, gpu_out);
    DWORD count = (GetTickCount() - tick);
    float sum = MatrixSum(gpu_out);
    std::cout << " CPU:" << sum << " Tick: " << count << std::endl;
  }

  NN::MathTerm();

  getchar();
}


