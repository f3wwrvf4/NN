#include "NN_math.h"
#include <string.h>

namespace NN
{

//
// Matrix
//
Matrix::Matrix(int size_x, int size_y)
{
  m_row_size = size_x;
  m_col_size = size_y;
  if (m_row_size * m_col_size != 0) {
    m_buff = new float[m_row_size * m_col_size];
  } else {
    m_buff = 0;
  }
}
Matrix::~Matrix()
{
  delete[] m_buff;
  m_buff = 0;
}

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out)
{
  _ASSERT(m1.m_row_size == m2.m_col_size);

  const int col1 = m1.m_col_size;
  const int row1 = m1.m_row_size;
//  const int col2 = m2.m_col_size;
  const int row2 = m2.m_row_size;

  const float* p1;
  const float* p2;
  float* po = out.m_buff;

  for (int j = 0; j < col1; ++j) {
    for (int i = 0; i < row2; ++i) {
      *po = 0;
      p1 = m1.m_buff + j*row1;
      p2 = m2.m_buff + i;
      for (int ii = 0; ii < row1; ++ii) {
        *po += (*p1) * (*p2);
        ++p1;
        p2 += row2;
      }
      ++po;
    }
  }
}

void Mul(const Vector& vec, const Matrix& mat, Vector& out)
{
  float* ptr = mat.m_buff;
  for (int j = 0; j < mat.m_col_size; ++j) {
    out.m_buff[j] = 0;
    for (int i = 0; i < mat.m_row_size; ++i) {
      out.m_buff[j] += *ptr++ * vec.m_buff[i];
    }
  }
}

std::ostream& operator <<(std::ostream& ost, const Matrix& mat)
{
  float* ptr = mat.m_buff;
  for (int j = 0; j < mat.m_col_size; ++j) {
    for (int i = 0; i < mat.m_row_size; ++i) {
      ost << *ptr++ << " ";
    }
    ost << std::endl;
  }
  return ost;
}

}