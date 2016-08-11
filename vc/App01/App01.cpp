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

  Matrix a(2000, 3000);
  Matrix b(1000, 2000);

  std::srand(0xababcdcd);
  a.random();
  b.random();

  Matrix c(b.row(), a.col());

  DWORD tick = GetTickCount();
  Mul(a, b, c);
//  std::cout << c << std::endl;
  std::cout << "Mul... " << (GetTickCount() - tick) << std::endl; // 27578

  float sum = 0;
  for (int i = 0; i < c.row(); ++i) {
    for (int j = 0; j < c.col(); ++j)
    sum += c(i, j);
  }

  std::cout << sum << std::endl;  // -2790.61

  NN::MathTerm();

  getchar();
}


