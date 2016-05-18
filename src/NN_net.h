#pragma once

#include "NN_math.h"
#include <vector>

namespace NN
{

struct TrainData
{
  TrainData():
    input({}),
    output({})
  {}

  TrainData(const std::vector<float>& inp, const std::vector<float>& out):
    input(inp),
    output(out)
  {
  }

  const std::vector<float>& input;
  const std::vector<float>& output;
};

class Weight : public Matrix
{
public:
  static void calcLogisticMid(const Matrix& weight, const Matrix& input, Matrix& out, Matrix& differential)
  {
    const Matrix& m1 = weight;
    const Matrix& m2 = input;

    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == out.m_col_size);
    _ASSERT(m2.m_row_size == out.m_row_size);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1;
    const float* p2;
    float* po1 = out.m_buff;
    float* po2 = differential.m_buff;

    for (int j = 0; j < col1-1; ++j) {
      for (int i = 0; i < row2; ++i) {
        *po1 = 0;
        p1 = m1.m_buff + j*row1;
        p2 = m2.m_buff + i;
        for (int ii = 0; ii < row1; ++ii) {
          *po1 += (*p1) * (*p2);
          ++p1;
          p2 += row2;
        }
        *po1 = 1.0f / (1.0f + exp(-*po1));
        *po2 = *po1*(1.0f - *po1);
        ++po1;
        ++po2;
      }
    }
    for (int i = 0; i < row2; ++i) {
      *po1++ = 1.0f;
      *po2++ = 1.0f;
    }
  }

  static void calcLogisticOut(const Matrix& weight, const Matrix& input, Matrix& out)
  {
    const Matrix& m1 = weight;
    const Matrix& m2 = input;

    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == out.m_col_size);
    _ASSERT(m2.m_row_size == out.m_row_size);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1;
    const float* p2;
    float* po1 = out.m_buff;

    for (int j = 0; j < col1; ++j) {
      for (int i = 0; i < row2; ++i) {
        *po1 = 0;
        p1 = m1.m_buff + j*row1;
        p2 = m2.m_buff + i;
        for (int ii = 0; ii < row1; ++ii) {
          *po1 += (*p1) * (*p2);
          ++p1;
          p2 += row2;
        }
        *po1 = 1.0f / (1.0f + exp(-*po1));
        ++po1;
      }
    }
  }

  static void calcHyperbolicTangent(const Matrix& input, Matrix& out, Matrix& differential)
  {

  }
  static void calcReLU(const Matrix& input, Matrix& out, Matrix& differential)
  {

  }
};

class Network
{
  const int W_NUM;
  const int I_NUM;
  const int L_LAST;
  const int L_NUM;
  const int D_NUM;
  const int D_LAST;

  Matrix** weights;
  Vector** inputs;
  Vector** backs;
  Vector** deltas;
  Matrix** deltas_t;

public:
  Network(int*L, int l_num);
  ~Network();

  void train(const TrainData* td_tbl, int num);
  std::vector<float> eval(std::vector<float> input);

  void save(const char* fpath);
  static Network* CreateFromFile(const char* fpath);

  float sigmoid(float v)
  {
    return 1.0f / (1.0f + exp(-v));
  }

protected:
  void feedForward();
  void backPropagate(const TrainData& data);

}; // Network


}