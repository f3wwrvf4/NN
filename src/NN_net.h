#pragma once

#include "NN_math.h"
#include <vector>
#include <iostream>

namespace NN
{

struct TrainData
{
  TrainData() :
    input({}),
    output({})
  {
  }

  TrainData(const std::vector<float>& inp, const std::vector<float>& out) :
    input(inp),
    output(out)
  {
  }

  const std::vector<float>& input;
  const std::vector<float>& output;
};

struct Logistic
{
  static void calcMid(const Matrix& weight, const Matrix& input, Matrix& out, Matrix& differential)
  {
    const Matrix& m1 = weight;
    const Matrix& m2 = input;

    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == out.m_col_size);
    _ASSERT(m2.m_row_size == out.m_row_size);

    _ASSERT(out.m_col_size == differential.m_col_size);
    _ASSERT(out.m_row_size == differential.m_row_size);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1;
    const float* p2;
    float* po1 = out.m_buff;
    float* po2 = differential.m_buff;

    for (int j = 0; j < col1 - 1; ++j) {
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
        *po2 = *po1 * (1.0f - *po1);
        ++po1;
        ++po2;
      }
    }
    for (int i = 0; i < row2; ++i) {
      *po1++ = 1.0f;
      *po2++ = 1.0f;
    }

    _ASSERT(po1 - out.m_buff == out.col()*out.row());

    _ASSERT(po2 - differential.m_buff == differential.col()*differential.row());
  }
  static void calcOut(const Matrix& weight, const Matrix& input, Matrix& out)
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

    const float* p1 = 0;
    const float* p2 = 0;
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

    _ASSERT(po1 - out.m_buff == out.col()*out.row());
  }

  static void evalMid(const Matrix& weight, const Matrix& input, Matrix& out)
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

    for (int j = 0; j < col1 - 1; ++j) {
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
    for (int i = 0; i < row2; ++i) {
      *po1++ = 1.0f;
    }

    _ASSERT(po1 - out.m_buff == out.col()*out.row());
  }
  static void evalOut(const Matrix& weight, const Matrix& input, Matrix& out)
  {
    calcOut(weight, input, out);
  }

  static void calcHyperbolicTangent(const Matrix& input, Matrix& out, Matrix& differential)
  {

  }
  static void calcReLU(const Matrix& input, Matrix& out, Matrix& differential)
  {

  }
};


template <class TYPE>
struct Layer
{
  Layer* prev;
  Layer* next;

  Layer(int _batch_num, int _node1, int _node2) :
    input_trn(_node1, _batch_num),
    out(_batch_num, _node2),
    out_vec(1, _node2),
    out_trn(_node2, _batch_num),
    weight(_node1, _node2),
    weight_trn(_node2, _node1),
    delta(_batch_num, _node1),
    differ(_batch_num, _node2),
    rdw(_node1, _node2)
  {
    batch_num = _batch_num;
    node1 = _node1;
    node2 = _node2;
    weight.random();
    prev = next = 0;
  }
  ~Layer()
  {
  }

  int batch_num;
  int node1, node2;

  NN::Matrix input_trn;

  NN::Matrix out;
  NN::Matrix out_vec;
  NN::Matrix out_trn;

  NN::Matrix weight;
  NN::Matrix weight_trn;

  NN::Matrix delta;
  NN::Matrix differ;
  NN::Matrix rdw;

  const Matrix* forward(const Matrix* input)
  {
    if (next)
      TYPE::calcMid(weight, *input, out, differ);
    else
      TYPE::calcOut(weight, *input, out);

    input->t(input_trn);

    return &out;
  }
  const Matrix* back(const Matrix* out_delta)
  {
    if (prev) {
      (weight).t(weight_trn);
      calcDelta(weight_trn, *out_delta, prev->differ, delta);
    }

    const float eps = 0.01f;
    calcWeight(*out_delta, input_trn, -eps, weight);

    return &delta;
  }

  const Matrix* eval(const Matrix* input)
  {
    if (next)
      TYPE::evalMid(weight, *input, out_vec);
    else
      TYPE::evalOut(weight, *input, out_vec);
    return &out_vec;
  }

  static void calcDelta(const Matrix& m1, const Matrix& m2, const Matrix& differ, Matrix& delta)
  {
    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == delta.m_col_size);
    _ASSERT(m2.m_row_size == delta.m_row_size);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1;
    const float* p2;
    const float* df = differ.m_buff;
    float* po = delta.m_buff;

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
        *po *= *df;
        ++po;
        ++df;
      }
    }
  }
  static void calcWeight(const Matrix& m1, const Matrix& m2, float eps, Matrix& weight)
  {
    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == weight.m_col_size);
    _ASSERT(m2.m_row_size == weight.m_row_size);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1;
    const float* p2;
    float rdw;

    float* w = weight.m_buff;

    for (int j = 0; j < col1; ++j) {
      for (int i = 0; i < row2; ++i) {
        rdw = 0;
        p1 = m1.m_buff + j*row1;
        p2 = m2.m_buff + i;
        for (int ii = 0; ii < row1; ++ii) {
          rdw += (*p1) * (*p2);
          ++p1;
          p2 += row2;
        }
        *w += rdw * eps;
        ++w;
      }
    }
  }

  void save(std::ostream& ost) const;
  void load(std::istream& ist);
};



struct Network
{
  Network(int layer_num, const int* node_num, int batch_num);
  ~Network();
  void train(const Matrix& input, const Matrix& out);
  const Matrix& eval(const Matrix& input) const;

  void save(const char* fpath) const;
  void load(const char* fpath);
  static Network* create(const char* fpath);

  NN::Layer<Logistic>** layers;
  const int layer_num;
  int* node_num;
  const int batch_num;

};
}

