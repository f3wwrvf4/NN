
#include "NN_net.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

namespace NN
{
void LayerBase::save(std::ostream& ost) const
{
  ost << weight;
}

void LayerBase::load(std::istream& ist)
{
  ist >> weight;
}

Network::Network(int layer_num_, const int* node_num_, int batch_num_):
  layer_num(layer_num_),
  batch_num(batch_num_)
{
  node_num = new int[layer_num];
  for (int i = 0; i < layer_num; ++i) {
    node_num[i] = node_num_[i];
  }

  const int layers_len = layer_num - 1;
  const int last_layer = layer_num - 1;

  layers = new LayerBase*[layers_len];

  NN::LayerBase* prev = 0;
  for (int i = 0; i < layers_len; ++i) {
    const bool isLast = i == (layers_len - 1);
    const int node1 = node_num[i] + 1;
    const int node2 = isLast ? node_num[i + 1] : node_num[i + 1] + 1;
    layers[i] = new NN::Layer<NN::Logistic>(batch_num, node1, node2);
    layers[i]->prev = prev;
    if (prev) prev->next = layers[i];
    prev = layers[i];
  }
}

Network::~Network()
{
  const int layers_len = layer_num - 1;
  for (int i = 0; i < layers_len; ++i) {
    delete layers[i];
  }
  delete[] layers;

  delete[] node_num;
}

void Network::train(const Matrix& input, const Matrix& output)
{
  const int layers_len = layer_num - 1;
  const int last_layer = layer_num - 1;

  const int Node1 = node_num[0] + 1;
  const int NodeL = node_num[last_layer];

  //  NN::Matrix input(batch_num, Node1);
  NN::Matrix delta_o(batch_num, NodeL);
  NN::LayerBase* layer;

  float error = 0;
  int iTrain = 0;
  while (true) {
    ++iTrain;
    error = 0;

    layer = layers[0];
    const NN::Matrix* in = &input;
    for (int i = 0; i < layers_len; ++i) {
      in = layer->forward(in);
      layer = layer->next;
    }

    // error
    const Matrix& out = layers[layers_len - 1]->out;
    for (int i = 0; i < batch_num; ++i) {
      for (int j = 0; j < NodeL; ++j) {
        const float err = NN::Square(output(i, j) - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - output(i, j);
      }
    }

    if (error < 0.1f) {
      for (int i = 0; i < batch_num; ++i) {
        for (int j = 0; j < NodeL; ++j) {
          float t = output(i, j);
          float o = out(i, j);
//          std::cout << t << " - " << o << std::endl;
        }
      }
//      std::cout << iTrain << ":" << error << std::endl;
      std::cout << "› " << std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
      break;
    }else if((iTrain%2000)==0){

      std::cout << "~ "<< std::setw(6) << iTrain << ":" << std::setw(8) << std::left << error << std::endl;
//      std::cout << input;
//      std::cout << output;
//      std::cout << out;

      break;
    }


    // back
    layer = layers[layers_len - 1];
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
  _ASSERT(input.col() == node_num[0] + 1);
//  input(0, node_num[0]) = 1.0f;

  const NN::Matrix* in = &input;
  for (int i = 0; i < layer_num - 1; ++i) {
    in = layers[i]->eval(in);
  }
  return layers[layer_num-2]->out_vec;
}


void Network::save(const char* fpath) const
{
  std::ofstream ofs(fpath);

  // input
  ofs << layer_num << " ";
  for (int i = 0; i < layer_num; ++i) {
    int sz = node_num[i];
    ofs << sz << " ";
  }

  for (int i = 0; i < layer_num - 1; ++i) {
    layers[i]->save(ofs);
  }

  ofs << std::endl;
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

  std::ifstream ifs(fpath);
  ifs >> ival;
  if (ival != layer_num) return;

  for (int i = 0; i < layer_num; ++i) {
    int sz = node_num[i];
    ifs >> ival;
    if (ival != sz) return;
  }

  for (int i = 0; i < layer_num - 1; ++i) {
    layers[i]->load(ifs);
  }
}
}
