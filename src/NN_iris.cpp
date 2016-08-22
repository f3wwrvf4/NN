#include "NN_iris.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

namespace NN
{
void Iris::LoadTrainData(Content& data)
{
  LoadData(data, true);
}
void Iris::LoadTestData(Content& data)
{
  LoadData(data, false);
}

void Iris::LoadData(Content& data, bool isTrain)
{
  // image
  {
    ifstream ifs("../data/iris.txt", ios::in);
    assert(!ifs.fail());

    char c;
    char str[32];

    int line = 0;
    int count[2][3] = {};
    while (!ifs.eof()) {
      float f[DataSize] = {};
      float o[LabelSize] = {};
      ifs >> f[0] >> c >> f[1] >> c >> f[2] >> c >> f[3] >> c >> str;

      int id = line & 1;
      bool fail = false;
      if (strcmp(str, "setosa") == 0) {
        o[0] = 1.0f;
        ++count[id][0];
      } else if (strcmp(str, "versicolor") == 0) {
        o[1] = 1.0f;
        ++count[id][1];
      } else if (strcmp(str, "virginica") == 0) {
        o[2] = 1.0f;
        ++count[id][2];
      } else {
        fail = true;
      }
      if (!fail) {
        if (isTrain && (line & 1)) {
          data.push_back(f, o);
        } else {
          data.push_back(f, o);
        }
      }
      ++line;
    }
  }
}

}