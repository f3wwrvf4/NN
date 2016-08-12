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
    cudaError_t err;
    err = cudaMalloc((void **)&gpu_buff, buff_sz);
    if (err != cudaSuccess) {
      std::cerr << "cublass cudaMalloc Error\n";
    }
  }
  ~Helper()
  {
    clear();
  }
  void set(int size, const float* cpu_buff) const
  {
    cublasStatus_t status;
    status = cublasSetVector(size, sizeof(cpu_buff[0]), cpu_buff, 1, gpu_buff, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cublass setVector Error\n";
    }
  }
  void get(int size, float* cpu_buff)
  {
    cublasStatus_t status;
    status = cublasGetVector(size, sizeof(cpu_buff[0]), gpu_buff, 1, cpu_buff, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cublass getVector Error\n";
    }
  }
  void clear()
  {
    if (gpu_buff) {
      cudaFree(gpu_buff);
      gpu_buff = 0;
    }
  }

  static bool enable;
  static bool initialized;
  static cublasHandle_t handle;
  static int device;

  static void init()
  {
    initialized = false;
    device = 0;

    // create
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "CUBLAS create error" << std::endl;
      return;
    }

    initialized = true;
  }
  static void term()
  {
    int status;

    // shutdown
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "!!!! shutdown error (A)\n";
    }

    initialized = false;
  }

  static void Gemm(float alpha, const Matrix& m1, const Matrix& m2, float beta, const Matrix& m3, Matrix& out)
  {
    m1.helper->set(m1.row()*m1.col(), m1.m_buff);
    m2.helper->set(m2.row()*m2.col(), m2.m_buff);
    out.helper->set(m3.row()*m3.col(), m3.m_buff);

    cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m1.col(), m2.row(), m1.row(),
      &alpha,
      m1.helper->gpu_buff, m1.col(),
      m2.helper->gpu_buff, m2.col(),
      &beta,
      out.helper->gpu_buff, out.col()
      );
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "gemm error\n";
    }
    out.helper->get(out.row()*out.col(), out.m_buff);
  }

};
bool Matrix::Helper::enable = false;
bool Matrix::Helper::initialized = false;
cublasHandle_t Matrix::Helper::handle;
int Matrix::Helper::device = 0;


//
// Matrix
//
Matrix::Matrix()
{
  m_row_size = 0;
  m_col_size = 0;
  m_buff = 0;
  helper = 0;
}

Matrix::Matrix(int size_x, int size_y)
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

Matrix::Matrix(const Matrix& mat)
{
  m_row_size = mat.m_row_size;
  m_col_size = mat.m_col_size;
  const int size = m_row_size * m_col_size;
  if (size != 0) {
    m_buff = new float[size];
    memcpy(m_buff, mat.m_buff, sizeof(float) * size);
  } else {
    m_buff = 0;
  }
  helper = new Helper(size);
}

Matrix& Matrix::operator = (const Matrix& mat)
{
  clear();

  m_row_size = mat.m_row_size;
  m_col_size = mat.m_col_size;
  const int size = m_row_size * m_col_size;
  if (size != 0) {
    m_buff = new float[size];
    memcpy(m_buff, mat.m_buff, sizeof(float) * size);
  } else {
    m_buff = 0;
  }
  helper = new Helper(size);

  return *this;
}

void Matrix::set(int size_x, int size_y)
{
  clear();

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

Matrix::~Matrix()
{
  clear();
}

void Matrix::clear()
{
  m_row_size = m_col_size = 0;
  delete[] m_buff;
  m_buff = 0;
  if (helper) {
    helper->clear();
    delete helper;
    helper = 0;
  }
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

void Matrix::Mul(const Matrix& m1, const Matrix& m2, Matrix& out)
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

void Matrix::Hadamard(const Matrix& m1, const Matrix& m2, Matrix& out)
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

const Matrix& Matrix::Mul(float f, const Matrix& m2, Matrix& out)
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

void Matrix::Add(float alpha, const Matrix& m1, float beta, const Matrix& m2, Matrix& out)
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

void Matrix::Gemm(float alpha, const Matrix& m1, const Matrix& m2, float beta, const Matrix& m3, Matrix& out)
{
  _ASSERT(m1.m_row_size == m2.m_col_size);
  _ASSERT(m1.m_col_size == out.m_col_size);
  _ASSERT(m2.m_row_size == out.m_row_size);
  _ASSERT(m3.row() == out.row());
  _ASSERT(m3.col() == out.col());

  if (Matrix::Helper::initialized && Matrix::Helper::enable) {
    Matrix::Helper::Gemm(alpha, m1, m2, beta, m3, out);
  } else {
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
        out(i, j) = (alpha * prod + beta * m3(i, j));
      }
    }
  }
}

void Matrix::Apply(const Matrix& m1, float(*func)(float), Matrix& out)
{
  float* pi = m1.m_buff;
  float* po = out.m_buff;
  const int sz = out.col()*out.row();
  for (int i = 0; i < sz; ++i) {
    *po++ = func(*pi++);
  }
}

std::ostream& operator << (std::ostream& ost, const Matrix& mat)
{
  ost << mat.row() << " " << mat.col() << std::endl;
  for (int j = 0; j < mat.col(); ++j) {
    for (int i = 0; i < mat.row(); ++i) {
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

std::istream& operator >> (std::istream& ist, Matrix& mat)
{
  int row, col;

  mat.clear();

  ist >> row >> col;
  mat.set(row, col);

  for (int j = 0; j < mat.col(); ++j) {
    for (int i = 0; i < mat.row(); ++i) {
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
  Matrix::Helper::init();
  Matrix::Helper::enable = true;
}

void MathTerm()
{
  Matrix::Helper::term();
}

void HelperEnable(bool flag)
{
  Matrix::Helper::enable = flag;
}


}