#pragma once

#include <iostream>
#include <vector>

#define ARRAY_NUM(_x) (sizeof(_x)/(sizeof(_x[0])))

namespace NN
{

class Vector;

class Matrix
{
public:
  Matrix();
  Matrix(int size_x, int size_y);
  Matrix(const Matrix& mat);
  ~Matrix();

  int row() const { return m_row_size; }
  int col() const { return m_col_size; }

  void clear()
  {
    m_row_size = m_col_size = 0;
    delete[] m_buff;
    m_buff = 0;
  }

  float& operator()(int x, int y)
  {
    return m_buff[y*row() + x];
  }
  float operator()(int x, int y) const
  {
    return m_buff[y*row() + x];
  }

  Matrix t() const;

  friend std::ostream& operator <<(std::ostream& ost, const Matrix& mat);
  friend std::istream& operator >>(std::istream& ist, Matrix& mat);
  friend void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
//  friend void Mul(const Vector& vec, const Matrix& mat, Vector& out);

protected:
  float* m_buff;
  int m_row_size;
  int m_col_size;
};

class Vector: public Matrix
{
public:
  Vector():
    Matrix(0,0)
  { }
  Vector(int size):
    Matrix(size, 1)
  {
  }

  int size() const { return row(); }

  float& operator()(int x)
  {
    return Matrix::operator()(x, 0);
  }
  float operator()(int x) const
  {
    return Matrix::operator()(x, 0);
  }

  friend void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
//  friend void Mul(const Vector& vec, const Matrix& mat, Vector& out);

  std::vector<float> vec() const
  {
    size_t sz = size();
    std::vector<float> v(sz);
    for (int i = 0; i < sz; ++i) {
      v[i] = m_buff[i];
    }
    return v;
  }
};

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
//void Mul(const Vector& vec, const Matrix& mat, Vector& out);

}