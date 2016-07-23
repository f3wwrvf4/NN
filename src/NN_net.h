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
  static void calcOut(const Matrix& weight, const Matrix& bias, const Matrix& input, Matrix& out)
  {
    const Matrix& m1 = weight;
    const Matrix& m2 = input;

    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == out.m_col_size);
    _ASSERT(m2.m_row_size == out.m_row_size);

    NN::Mul(m1, m2, out);
    NN::Add(out, bias, out);

    float* po1 = out.m_buff;
    const int sz = out.m_col_size * out.m_row_size;
    for (int i = 0; i < sz; ++i) {
      *po1 = 1.0f / (1.0f + exp(-*po1));
      ++po1;
    }
  }
  static void calcDiff(const Matrix& out, Matrix& diff)
  {
    _ASSERT(out.m_row_size == diff.m_row_size);
    _ASSERT(out.m_col_size == diff.m_col_size);

    const float* po1 = out.m_buff;
    float* po2 = diff.m_buff;
    const int sz = out.m_col_size * out.m_row_size;
    for (int i = 0; i < sz; ++i) {
      *po2 = *po1 * (1.0f - *po1);
      ++po1;
      ++po2;
    }
  }

};


struct SoftMax
{
  static void calcOut(const Matrix& weight, const Matrix& input, Matrix& out)
  {
    const Matrix& m1 = weight;
    const Matrix& m2 = input;

    _ASSERT(m1.m_row_size == m2.m_col_size);
    _ASSERT(m1.m_col_size == out.m_col_size);
    _ASSERT(m2.m_row_size == out.m_row_size);

    Mul(weight, input, out);

    const int col1 = m1.m_col_size;
    const int row1 = m1.m_row_size;
    //  const int col2 = m2.m_col_size;
    const int row2 = m2.m_row_size;

    const float* p1 = 0;
    const float* p2 = 0;
    float* po1 = out.m_buff;

    for (int j = 0; j < col1; ++j) {
      // calc weight value
      float* pws = po1; // weight sum
      float max_val = 0;
      for (int i = 0; i < row2; ++i) {
        if (fabs(max_val) < fabs(*pws)) {
          max_val = *pws;
        }
        ++pws;
      }
      float expsum = 0;
      float* ep = po1;
      for (int i = 0; i < row2; ++i) {
        const float e = exp(*ep - max_val);
        *ep = e;
        expsum += e;
        ++ep;
      }

      for (int i = 0; i < row2; ++i) {
        *(po1) /= expsum;
        ++po1;
      }
    }

    _ASSERT(po1 - out.m_buff == out.col()*out.row());
  }
  static void calcDiff(const Matrix& out, Matrix& diff)
  {
    _ASSERT(0);
  }

  static void evalMid(const Matrix& weight, const Matrix& input, Matrix& out)
  {
    _ASSERT(0);
  }
  static void evalOut(const Matrix& weight, const Matrix& input, Matrix& out)
  {
    calcOut(weight, input, out);
  }
};


struct LayerBase
{
  LayerBase* prev;
  LayerBase* next;

  LayerBase(int _batch_num, int _node1, int _node2) :
    input(_batch_num, _node1),
    input_trn(_node1, _batch_num),
    out(_batch_num, _node2),
    out_vec(1, _node2),
    out_trn(_node2, _batch_num),
    weight(_node1, _node2),
    weight_trn(_node2, _node1),
    bias_vec(1, _node2),
    bias_mat(_batch_num, _node2),
    delta(_batch_num, _node1),
    differ(_batch_num, _node2),
    rdw(_node1, _node2)
  {
    batch_num = _batch_num;
    node1 = _node1;
    node2 = _node2;
    weight.random();
    bias_vec.random();
    prev = next = 0;
  }
  ~LayerBase()
  {
  }

  int batch_num;
  int node1, node2;

  NN::Matrix input;
  NN::Matrix input_trn;

  NN::Matrix out;
  NN::Matrix out_vec;
  NN::Matrix out_trn;

  NN::Matrix weight;
  NN::Matrix weight_trn;

  NN::Matrix bias_vec;
  NN::Matrix bias_mat;

  NN::Matrix delta;
  NN::Matrix differ;
  NN::Matrix rdw;

  static void calcDelta(const Matrix& m1, const Matrix& m2, const Matrix& differ, Matrix& delta);
  static void calcWeight(const Matrix& m1, const Matrix& m2, Matrix& rdw, float eps, Matrix& weight, Matrix& bias);

  void save(std::ostream& ost) const;
  void load(std::istream& ist);

  virtual const Matrix* forward(const Matrix*) = 0;
  virtual const Matrix* back(const Matrix*) = 0;
  virtual const Matrix* eval(const Matrix*) = 0;
};

template <class TYPE>
struct Layer : public LayerBase
{
  Layer(int _batch_num, int _node1, int _node2) :
    LayerBase(_batch_num, _node1, _node2)
  {}

  const Matrix* forward(const Matrix* in)
  {
    memcpy(input.m_buff, in->m_buff, in->row()*in->col()*sizeof(float));

    float*po = bias_mat.m_buff;
    for (int i = 0; i < bias_vec.col(); ++i) {
      const float val = bias_vec(0, i);
      for (int j = 0; j < bias_mat.row(); ++j) {
        *po++ = val;
      }
    }

    TYPE::calcOut(weight, bias_mat, input, out);
    input.t(input_trn);
    return &out;
  }
  const Matrix* back(const Matrix* out_delta)
  {
    if (prev) {
      TYPE::calcDiff(prev->out, prev->differ);
      (weight).t(weight_trn);
      calcDelta(weight_trn, *out_delta, prev->differ, delta);
    }

    const float eps = 0.1f/(float)batch_num;
    calcWeight(*out_delta, input_trn, rdw, -eps, weight, bias_vec);

    return &delta;
  }

  const Matrix* eval(const Matrix* input)
  {
    TYPE::calcOut(weight, bias_mat, *input, out_vec);
    return &out_vec;
  }
};

struct Network
{
  typedef enum LayerType
  {
    LogisticLayer,
    SoftMaxLayer,
  } LayerType;
  typedef struct InitParam
  {
    int node_num[2];  // in, out
    LayerType layer_type;
  } InitParam;

  Network(int layer_num, const InitParam* node_param, int batch_num);
  ~Network();
  void train(const Matrix& input, const Matrix& out);
  const Matrix& eval(const Matrix& input) const;

  void save(const char* fpath) const;
  void load(const char* fpath);
  static Network* create(const char* fpath);

  NN::LayerBase** layers;
  const int layer_num;
//  int* node_num;
  const int batch_num;

};
}

