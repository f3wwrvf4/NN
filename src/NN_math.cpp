#include "NN_math.h"
#include <string.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cublas_v2.h>

namespace NN
{
//
// Matrix
//
Matrix::Matrix()
{
  m_row_size = 0;
  m_col_size = 0;
  m_buff = 0;
}

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
Matrix::Matrix(const Matrix& mat)
{
  m_row_size = mat.m_row_size;
  m_col_size = mat.m_col_size;
  if (m_row_size * m_col_size != 0) {
    const size_t sz = m_row_size * m_col_size;
    m_buff = new float[sz];
    memcpy(m_buff, mat.m_buff, sizeof(float) * sz);
  } else {
    m_buff = 0;
  }
}

Matrix::~Matrix()
{
  delete[] m_buff;
  m_buff = 0;
}

void Matrix::clear()
{
  m_row_size = m_col_size = 0;
  delete[] m_buff;
  m_buff = 0;
}

void Matrix::random()
{
  for (int j = 0; j < col(); ++j)
    for (int i = 0; i < row(); ++i)
      (*this)(i, j) = rand() / (float)RAND_MAX - 0.5f;
}

float& NN::Matrix::operator()(int x, int y)
{
  _ASSERT(0<=x);
  _ASSERT(0<=y);
  _ASSERT(x<row());
  _ASSERT(y<col());
  return m_buff[x*col() + y];
}
float NN::Matrix::operator()(int x, int y) const
{
  _ASSERT(0<=x);
  _ASSERT(0<=y);
  _ASSERT(x<row());
  _ASSERT(y<col());
  return m_buff[x*col() + y];
}


 void Matrix::t(Matrix& mat) const
{
  for (int i = 0; i < m_row_size; ++i) {
    for (int j = 0; j < m_col_size; ++j) {
      mat(j, i) = (*this)(i, j);
    }
  }
}

std::vector<float> Vector::vec() const
{
  size_t sz = size();
  std::vector<float> v(sz);
  for (int i = 0; i < sz; ++i) {
    v[i] = m_buff[i];
  }
  return v;
}

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out)
{
  _ASSERT(m1.m_row_size == m2.m_col_size);
  _ASSERT(m1.m_col_size == out.m_col_size);
  _ASSERT(m2.m_row_size == out.m_row_size);

  const int col1 = m1.m_col_size;
  const int row1 = m1.m_row_size;
  //  const int col2 = m2.m_col_size;
  const int row2 = m2.m_row_size;

  for (int i = 0; i < row2; ++i) {
    for (int j = 0; j < col1; ++j) {
      float& o = out(i, j);
      o = 0;
      for (int ii = 0; ii < row1; ++ii) {
        const float a = m1(ii, j);
        const float b = m2(i, ii);
        o += a * b;
      }
    }
  }
}

void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out)
{
  _ASSERT(m1.m_row_size == m2.m_row_size);
  _ASSERT(m1.m_row_size == out.m_row_size);
  _ASSERT(m1.m_col_size == m2.m_col_size);
  _ASSERT(m1.m_col_size == out.m_col_size);

  const float* p1 = m1.m_buff;
  const float* p2 = m2.m_buff;
  float* po = out.m_buff;

  const int sz = m1.col()*m2.row();
  for (int i = 0; i < sz; ++i) {
    *po++ = *p1++ * *p2++;
  }
}

const Matrix& Mul(float f, const Matrix& m2, Matrix& out)
{
  _ASSERT(m2.m_row_size == out.m_row_size);
  _ASSERT(m2.m_col_size == out.m_col_size);

  const float* p2 = m2.m_buff;
  float* po = out.m_buff;

  const int sz = m2.col()*m2.row();
  for (int i = 0; i < sz; ++i) {
    *po++ = f * *p2++;
  }
  return out;
}
void Add(const Matrix& m1, const Matrix& m2, Matrix& out)
{
  _ASSERT(m1.m_row_size == m2.m_row_size);
  _ASSERT(m1.m_row_size == out.m_row_size);
  _ASSERT(m1.m_col_size == m2.m_col_size);
  _ASSERT(m1.m_col_size == out.m_col_size);

  const float* p1 = m1.m_buff;
  const float* p2 = m2.m_buff;
  float* po = out.m_buff;

  const int sz = m1.col()*m2.row();
  for (int i = 0; i < sz; ++i) {
    *po++ = *p1++ + *p2++;
  }
}

std::ostream& operator <<(std::ostream& ost, const Matrix& mat)
{
  ost << mat.m_row_size << " " << mat.m_col_size << std::endl;
  for (int j = 0; j < mat.m_col_size; ++j) {
    for (int i = 0; i < mat.m_row_size; ++i) {
      ost
        << std::setw(7)
        << std::right
        << std::fixed
        << std::setprecision(4)
        << mat(i,j) << " ";
    }
    ost << std::endl;
  }

  return ost;
}

std::istream& operator >>(std::istream& ist, Matrix& mat)
{
  int row, col;

  mat.clear();

  ist >> row >> col;
  mat.m_row_size = row;
  mat.m_col_size = col;

  mat.m_buff = new float[row*col];

  for (int j = 0; j < mat.m_col_size; ++j) {
    for (int i = 0; i < mat.m_row_size; ++i) {
      ist >> mat(i, j);
    }
  }
  return ist;
}

float Square(float val)
{
  return val*val; 
}


}