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

  inline int row() const { return m_row_size; }
  inline int col() const { return m_col_size; }

  inline float& operator()(int x, int y);
  inline float operator()(int x, int y) const;

  void t(Matrix&) const;

  void random();

  friend std::ostream& operator <<(std::ostream& ost, const Matrix& mat);
  friend std::istream& operator >>(std::istream& ist, Matrix& mat);
  friend void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
  friend void Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out);
  friend void Gemm(float alpha, const Matrix& m1, const Matrix& m2, float beta, const Matrix& m3, Matrix& out);

  friend void Apply(const Matrix& m1, float(*func)(float), Matrix& out);

protected:
  float* m_buff;
  int m_row_size;
  int m_col_size;

  struct Helper;
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
  friend void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
  friend const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
  friend void Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out);
  friend void Gemm(float alpha, const Matrix& m1, const Matrix& m2, const Matrix& m3, Matrix& out);
};

void Mul(const Matrix& m1, const Matrix& m2, Matrix& out);
void Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out);
const Matrix& Mul(float f, const Matrix& m2, Matrix& out);
void Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out);
void Gemm(float alpha, const Matrix& m1, const Matrix& m2, const Matrix& m3, Matrix& out);

float Square(float val);


void MathInit();
void MathTerm();

}