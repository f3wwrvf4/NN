
#include "NN_net.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

namespace NN
{
void LayerBase::calcDelta(const Matrix& m1, const Matrix& m2, const Matrix& differ, Matrix& delta)
{
  _ASSERT(m1.row() == m2.col());
  _ASSERT(m1.col() == delta.col());
  _ASSERT(m2.row() == delta.row());

  Matrix::Mul(m1, m2, delta);
  Matrix::Hadamard(delta, differ, delta);
}

void LayerBase::calcWeight(const Matrix& m1, const Matrix& m2, Matrix& rdw, float eps, Matrix& weight, Matrix& bias)
{
  _ASSERT(m1.row() == m2.col());
  _ASSERT(m1.col() == weight.col());
  _ASSERT(m2.row() == weight.row());

  // weight
  {
#if 1
    Matrix::Gemm(1.0f, m1, m2, 0.0f, rdw, rdw);
    Matrix::Add(1.0f, weight, eps, rdw, weight);
#else
    Mul(m1, m2, rdw);
    float* po1 = weight.m_buff;
    const float* po2 = rdw.m_buff;
    const int sz = weight.col() * weight.row();
    for (int i = 0; i < sz; ++i) {
      *po1 += *po2 * eps;
      ++po1;
      ++po2;
    }
#endif
  }

  _ASSERT(bias.col() == m1.col());
  _ASSERT(bias.row() == 1);
  // bias
  {
#if 1
    for (int i = 0; i < m1.col(); ++i) {
      float val = 0;
      for (int j = 0; j < m1.row(); ++j) {
        val += m1(j, i);
      }
      bias(0, i) += val * eps;
    }
#else
    float* po = bias.m_buff;
    float* pb = m1.m_buff;
    for (int i = 0; i < m1.col(); ++i) {
      float val = 0;
      for (int j = 0; j < m1.row(); ++j) {
        val += *pb++;
      }
      *po++ += val*eps;
    }
#endif
  }
}

void LayerBase::save(std::ostream& ost) const
{
  ost << weight << bias_vec;
}

void LayerBase::load(std::istream& ist)
{
  ist >> weight >> bias_vec;
}

Network::Network(int layer_num_, const InitParam* init_param, int batch_num_) :
  layer_num(layer_num_),
  batch_num(batch_num_)
{
  layers = new LayerBase*[layer_num];

  NN::LayerBase* prev = 0;
  for (int i = 0; i < layer_num; ++i) {
    const bool isLast = i == (layer_num - 1);
    const int node1 = init_param[i].node_num[0];
    const int node2 = init_param[i].node_num[1];
    switch (init_param[i].layer_type) {
    case NN::Network::LogisticLayer:
      layers[i] = new NN::Layer<NN::Logistic>(batch_num, node1, node2);
      break;
    case SoftMaxLayer:
      layers[i] = new NN::Layer<NN::SoftMax>(batch_num, node1, node2);
      break;
    }
    layers[i]->prev = prev;
    if (prev) prev->next = layers[i];
    prev = layers[i];
  }
}

Network::~Network()
{
  for (int i = 0; i < layer_num; ++i) {
    delete layers[i];
  }
  delete[] layers;
}

void Network::train(const Matrix& input, const Matrix& output)
{
  const int last_layer = layer_num - 1;

  const int Node1 = layers[0]->node1;
  const int NodeL = layers[last_layer]->node2;

  NN::Matrix delta_o(batch_num, NodeL);
  NN::LayerBase* layer;

  float error = 0;
  int iTrain = 0;

//  while(true)
  {
    ++iTrain;
    error = 0;

    layer = layers[0];
    const NN::Matrix* in = &input;
    for (int i = 0; i < layer_num; ++i) {
      in = layer->forward(in);
      layer = layer->next;
    }

    // error
    const Matrix& out = layers[layer_num-1]->out;
    for (int i = 0; i < batch_num; ++i) {
      for (int j = 0; j < NodeL; ++j) {
        const float err = NN::Square(output(i, j) - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - output(i, j);
      }
    }

#if 0
    if (error < 0.1f) {
      for (int i = 0; i < batch_num; ++i) {
        for (int j = 0; j < NodeL; ++j) {
          float t = output(i, j);
          float o = out(i, j);
//          std::cout << t << " - " << o << std::endl;
        }
      }
//      std::cout << iTrain << ":" << error << std::endl;
//      std::cout << "› " << std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
      break;
    }else if((iTrain%2000)==0){

//      std::cout << "~ "<< std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
//      std::cout << input;
//      std::cout << output;
//      std::cout << out;

      break;
    }
#endif

    // back
    layer = layers[layer_num-1];
    const Matrix* delta = &delta_o;
    while (layer) {
      delta = layer->back(delta);
      layer = layer->prev;
    }
  }
}

const Matrix& Network::eval(const Matrix& input) const
{

  _ASSERT(input.row() == 1);
//  _ASSERT(input.col() == node_num[0] + 1);
//  input(0, node_num[0]) = 1.0f;

  const NN::Matrix* in = &input;
  for (int i = 0; i < layer_num; ++i) {
    in = layers[i]->forward(in);
  }

  return *in;
}

template <typename T>
std::ostream& Write(std::ostream& ofs, T val)
{
  ofs.write((char*)&val, sizeof(T));
  return ofs;
}

template <typename T>
std::istream& Read(std::istream& ifs, T& val)
{
  ifs.read((char*)&val, sizeof(T));
  return ifs;
}

void Network::save(const char* fpath) const
{
  std::ofstream ofs(fpath, std::ios::binary);

  // input
  Write(ofs, layer_num);
  for (int i = 0; i < layer_num; ++i) {
    int sz = layers[i]->node1;
    Write(ofs, sz);
  }
  {
    int sz = layers[layer_num-1]->node2;
    Write(ofs, sz);
  }
  for (int i = 0; i < layer_num; ++i) {
    layers[i]->save(ofs);
  }
}

Network* Network::create(const char* fpath)
{
#if 0
  int layer_num;
  int* node_num;
  int batch_num;

  std::ifstream ifs(fpath);
  ifs >> layer_num;
  node_num = new int[layer_num];
  for (int i = 0; i < layer_num; ++i) {
    ifs >> node_num[i];
  }

  Network* net = new Network(layer_num, node_num, batch_num);
  for (int i = 0; i < layer_num - 1; ++i) {
    net->layers[i]->load(ifs);
  }
  return net;
#else
  return 0;
#endif
}

void Network::load(const char* fpath)
{
  int ival;

  std::ifstream ifs(fpath, std::ios::binary);
  Read(ifs, ival);
  if (ival != layer_num) return;

  for (int i = 0; i < layer_num; ++i) {
    const int sz = layers[i]->node1;
    Read(ifs, ival);
    if (ival != sz) return;
  }

  {
    const int sz = layers[layer_num-1]->node2;
    Read(ifs, ival);
    if (ival != sz) return;
  }

  for (int i = 0; i < layer_num; ++i) {
    layers[i]->load(ifs);
  }
}
}
