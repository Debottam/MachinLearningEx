// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <arpa/inet.h>

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads( 1 );
  sessionOptions.SetGraphOptimizationLevel( ORT_ENABLE_BASIC );
  const char* model_path = "/afs/cern.ch/work/d/dbakshig/public/Onnx/deepCalo.onnx";
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Session session(env, model_path, sessionOptions);
  std::vector<int64_t> input_node_dims;
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  for( std::size_t i = 0; i < num_input_nodes; i++ ) {
     // print input node names
     char* input_name = session.GetInputName(i, allocator);
     std::cout<<"Input "<<i<<" : "<<" name= "<<input_name<<std::endl;
     input_node_names[i] = input_name;
 // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout<<"Input "<<i<<" : "<<" type= "<<type<<std::endl;

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    std::cout<<"Input "<<i<<" : num_dims= "<<input_node_dims.size()<<std::endl;
    for (int j = 0; j < input_node_dims.size(); j++){
      if(input_node_dims[j]<0)
       input_node_dims[j] =1;
      std::cout<<"Input"<<i<<" : dim "<<j<<"= "<<input_node_dims[j]<<std::endl;
 }
}

//output nodes
  std::vector<int64_t> output_node_dims;
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);

  for( std::size_t i = 0; i < num_output_nodes; i++ ) {
     // print output node names
     char* output_name = session.GetOutputName(i, allocator);
     std::cout<<"Output "<<i<<" : "<<" name= "<<output_name<<std::endl;
     output_node_names[i] = output_name;
 // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout<<"Output "<<i<<" : "<<" type= "<<type<<std::endl;

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    std::cout<<"Output "<<i<<" : num_dims= "<<output_node_dims.size()<<std::endl;
    for (int j = 0; j < output_node_dims.size(); j++){
      if(output_node_dims[j]<0)
       output_node_dims[j] =1;
      std::cout<<"Output"<<i<<" : dim "<<j<<"= "<<output_node_dims[j]<<std::endl;
 }
}


return 0;
}
