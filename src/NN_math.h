#pragma once

#include <vector>

#define ARRAY_NUM(_x) (sizeof(_x)/(sizeof(_x[0])))

namespace NN
{

class Vector;

struct Matrix
{
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

  void t(Matrix&) const;

  void random()
  {
    size_t sz = m_row_size * m_col_size;
    for (int i = 0; i < sz; ++i)
      m_buff[i] = rand() / (float)RAND_MAX - 0.5f;
  }

  friend std::ostream& operator <<(std::ostream& ost, const Matrix& mat);
  friend std::istream& operator >>(std::istream& ist, Matrix& mat);
  friend void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
  friend void Add(const Matrix& m1, const Matrix& m2, Matrix& out);

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
  friend void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
  friend void Add(const Matrix& m1, const Matrix& m2, Matrix& out);
};

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
void Add(const Matrix& m1, const Matrix& m2, Matrix& out);


float Square(float val);
}