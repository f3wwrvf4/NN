
#include "stdafx.h"
#include "CppUnitTest.h"
#include "NN_math.h"

#include <sstream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{		
	TEST_CLASS(UnitTest1)
	{
	public:


      TEST_METHOD(Test001)
      {
        NN::Matrix m1(3, 2);
        NN::Matrix m2(3, 3);

        m1(0, 0) = 6; m1(1, 0) = 0; m1(2, 0) = 3;
        m1(0, 1) = 7; m1(1, 1) = 1; m1(2, 1) = 4;

        m1(0, 0) = 7; m1(1, 0) = 6; m1(2, 0) = 4;
        m1(0, 1) = 1; m1(1, 1) = 2; m1(2, 1) = 2;
        m1(0, 2) = 5; m1(1, 2) = 3; m1(2, 2) = 1;


        NN::Matrix o(3, 2);
        NN::Matrix::Mul(m1, m2, o);

        std::stringstream ss;
        ss << o;

        OutputDebugStringA(ss.str().c_str());
      }


      TEST_METHOD(Test002)
      {
        NN::Vector vec(2);
        NN::Matrix mat(2, 3);

        vec(0) = 3;
        vec(1) = 1;

        mat(0, 0) = 20;
        mat(0, 1) = 30;
        mat(0, 2) = 40;
        mat(1, 0) = 50;
        mat(1, 1) = 60;
        mat(1, 2) = 70;

        NN::Vector out(mat.col());
        NN::Matrix::Mul(vec, mat, out);


        float val = 1.0f;
        Assert::AreEqual(1.0f, val);

      }
	};
}