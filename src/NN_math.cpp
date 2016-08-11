#include "NN_math.h"
#include <string.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>




namespace NN
{
struct Matrix::Helper
{
  float* gpu_buff;
  Helper(int size)
  {
    const int buff_sz = size * sizeof(gpu_buff[0]);
//    cudaMalloc((void **)&gpu_buff, buff_sz);
  }
  void set(int size, float* cpu_buff)
  {
//    cublasSetVector(size, sizeof(cpu_buff[0]), cpu_buff, 1, gpu_buff, 1);
  }
  void clear()
  {
  }



  static bool initialized;
  static cublasHandle_t handle;
  static int device;

  static void init()
  {
#if 0
    int status;

    initialized = false;
    device = 0;

    // create
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "CUBLAS create error" << std::endl;
    }

#endif
  }
  static void term()
  {

  }

};


//
// Matrix
//
Matrix::Matrix():
  helper(0)
{
  m_row_size = 0;
  m_col_size = 0;
  m_buff = 0;
}

Matrix::Matrix(int size_x, int size_y):
  helper(0)
{
  m_row_size = size_x;
  m_col_size = size_y;
  const int size = m_row_size * m_col_size;
  if (size != 0) {
    m_buff = new float[size];
  } else {
    m_buff = 0;
  }

  helper = new Helper(size);
}

Matrix::Matrix(const Matrix& mat):
  helper(0)
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
  helper->clear();
  delete helper;
}

void Matrix::clear()
{
  m_row_size = m_col_size = 0;
  delete[] m_buff;
  m_buff = 0;
  helper->clear();
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
#if 1
void Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out)
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
    *po++ = alpha * *p1++ + beta * *p2++;
  }
}
#endif
void Gemm(float alpha, const Matrix& m1, const Matrix& m2, float beta, const Matrix& m3, Matrix& out)
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
      float prod = 0;
      for (int ii = 0; ii < row1; ++ii) {
        const float a = m1(ii, j);
        const float b = m2(i, ii);
        prod += a * b;
      }
      out(i, j) = alpha * prod + beta * m3(i,j);
    }
  }
}

void Apply(const Matrix& m1, float(*func)(float), Matrix& out)
{
  float* pi = m1.m_buff;
  float* po = out.m_buff;
  const int sz = out.col()*out.row();
  for (int i = 0; i < sz; ++i) {
    *po++ = func(*pi++);
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

void MathInit()
{

}

void MathTerm()
{

}

}