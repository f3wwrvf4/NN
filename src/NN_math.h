#pragma once

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

  void clear();

  int row() const { return m_row_size; }
  int col() const { return m_col_size; }

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

protected:
  float* m_buff;
  int m_row_size;
  int m_col_size;
};

class Vector: public Matrix
{
public:
  Vector():
    Matrix()
  {}
  Vector(int size):
    Matrix(size, 1)
  {}
  std::vector<float> vec() const;

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
};

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);

}