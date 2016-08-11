// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>

#include <NN_net.h>
#include <NN_math.h>
#include <NN_mnist.h>
#include <NN_iris.h>

int main()
{
  using namespace NN;

  NN::MathInit();

#if 1
  Matrix a(100, 100);
  Matrix b(100, 100);
#else
  Matrix a(2000, 3000);
  Matrix b(1000, 2000);
#endif

  std::srand(0xababcdcd);
  a.random();
  b.random();

  Matrix c(b.row(), a.col());
  Matrix d(b.row(), a.col());

  DWORD tick = GetTickCount();
//  Matrix::Mul(a, b, c);

  Matrix::Gemm(1.0f, a, b, 0.0f, d, c);

//  std::cout << c << std::endl;
  std::cout << "Mul... " << (GetTickCount() - tick) << std::endl; // 125 / 27578

  float sum = 0;
  for (int i = 0; i < c.row(); ++i) {
    for (int j = 0; j < c.col(); ++j) {
      sum += c(i, j);
    }
  }

  std::cout << sum << std::endl;  // 45.5688 / -2790.61

  NN::MathTerm();

  getchar();
}


