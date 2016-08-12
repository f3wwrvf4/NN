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

  void set(int size_x, int size_y);
  void clear();

  inline int row() const { return m_row_size; }
  inline int col() const { return m_col_size; }

  inline float& operator()(int x, int y);
  inline float operator()(int x, int y) const;

  void t(Matrix&) const;
  void random();

  static void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
  static void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
  static const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
  static void Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out);
  static void Gemm(float alpha, const Matrix& m1, const Matrix& m2, float beta, const Matrix& m3, Matrix& out);
  static void Apply(const Matrix& m1, float(*func)(float), Matrix& out);

  struct Helper;

protected:
  float* m_buff;
  int m_row_size;
  int m_col_size;

  Helper* helper;
};

class Vector : public Matrix
{
public:
  Vector() :
    Matrix()
  {
  }
  Vector(int size) :
    Matrix(size, 1)
  {
  }
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
};

float Square(float val);

std::ostream& operator <<(std::ostream& ost, const Matrix& mat);
std::istream& operator >>(std::istream& ist, Matrix& mat);

void MathInit();
void MathTerm();

}