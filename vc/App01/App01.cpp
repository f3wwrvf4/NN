// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "NN_math.h"
#include <iostream>

namespace NN
{
class Network
{
public:
  Network(const Vector& vec)
  {
    for (int i = 0; i < vec.size(); ++i) {

    }
  }
};

}

int main()
{
  NN::Network net(NN::Vector(4));

  {
    NN::Matrix m1(3, 2);
    NN::Matrix m2(2, 3);

    m1(0, 0) = 1; m1(1, 0) = 2; m1(2, 0) = 3;
    m1(0, 1) = 4; m1(1, 1) = 5; m1(2, 1) = 6;
    //    m1(0, 2) = 5; m1(1, 2) = 6; m1(2, 2) = 3;

    m2(0, 0) = 1; m2(1, 0) = 2;
    m2(0, 1) = 3; m2(1, 1) = 4;
    m2(0, 2) = 5; m2(1, 2) = 6;

    NN::Matrix o(m1.row(), m2.col());
    NN::Mul(m2, m1, o);

    std::cout << o;
  }


  getchar();
  return 0;
}

