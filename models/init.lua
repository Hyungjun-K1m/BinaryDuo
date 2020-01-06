--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--


require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = require('models/'..opt.netType)(opt)
      model = torch.load(modelPath):type(opt.tensorType)
      model.__memoryOptimized = nil
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      model = require('models/'..opt.netType)(opt)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):type(opt.tensorType)
      model.__memoryOptimized = nil
   elseif opt.deploy ~= 'none' then
      model = require('models/' .. opt.netType)(opt)
      print('=> Deploying model from '.. opt.deploy)
      local pretrained = torch.load(opt.deploy..'/model_best.t7')
      if opt.deployOpt == 0 then
         error('Please define deploy option!')
      elseif opt.deployOpt == 6 then
         print('=> decoupling coupled model...')
         model = model_decouple(model,pretrained,opt)
      else
         error('deploy option not supported!')
      end            
      model:type(opt.tensorType)
      if opt.cudnn == 'deterministic' then
         model:apply(function(m)
            if m.setMode then m:setMode(1,1,1) end
         end)
      end

      model:get(1).gradInput = nil
      print(model)

   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      print(opt.netType)
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):type(opt.tensorType)
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model, opt)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:type(opt.tensorType))
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   cutorch.setDevice(1)
   cutorch.setHeapTracking(true)
   if opt.nGPU > 1 then
    dpt = nn.DataParallelTable(1, true, true)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        dpt:add(model:clone():cuda(), i)
        -- dpt:add(model:clone():cuda(), i):threads(function()
        -- cudnn.fastest, cudnn.benchmark = fastest, benchmark
        -- end)  -- Use the ith GPU
    end
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark
    dpt:threads(function()
            local cudnn = require 'cudnn'
            require '../newLayers/QuantizedNeurons'
            require '../newLayers/cudnnQuantSpatialConvolution'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
    cutorch.setDevice(1)
    dpt.gradInput = nil
    model = dpt:type(opt.tensorType)
   end

   cutorch.setDevice(1)
   local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)

   return model, criterion
end

function M.shareGradInput(model, opt)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
end


function model_decouple(model,pretrained,opt)  
   if opt.netType == 'tinyVGG7_decoupled' then
      
      local function duplicateBN(from,to)
         local sz = pretrained:get(from).weight:size(1)
         model:get(to).running_mean[{{1,sz}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_var[{{1,sz}}]:copy(pretrained:get(from).running_var)
         model:get(to).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_var)
         model:get(to).weight[{{1,sz}}]:copy(pretrained:get(from).weight)
         model:get(to).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from).weight)
         model:get(to).bias[{{1,sz}}]:copy(pretrained:get(from).bias-0.25)
         model:get(to).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from).bias+0.25)
      end

      local function duplicateConv(from,to)

         local sz = pretrained:get(from).weight:size(2)
         model:get(to).weight[{{},{1,sz},{},{}}]:copy(pretrained:get(from).weight/2)
         model:get(to).weight[{{},{1+sz,sz*2},{},{}}]:copy(pretrained:get(from).weight/2)

      end
      

      model:get(1).weight:copy(pretrained:get(1).weight)
      duplicateBN(2,4)
      duplicateConv(5,7)
      duplicateBN(7,11)
      duplicateConv(10,14)
      duplicateBN(12,18)
      duplicateConv(15,21)
      duplicateBN(17,25)
      duplicateConv(20,28)
      duplicateBN(21,31)
      duplicateConv(24,34)
      model:remove(35)
      model:insert(pretrained:get(25),35)
      model:get(37).weight:copy(pretrained:get(27).weight)
      
   elseif opt.netType == 'alexnet_decoupled' then
      local function duplicateBN(from,to)
         local sz = pretrained:get(from).weight:size(1)
         model:get(to).running_mean[{{1,sz}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_var[{{1,sz}}]:copy(pretrained:get(from).running_var)
         model:get(to).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_var)
         model:get(to).weight[{{1,sz}}]:copy(pretrained:get(from).weight)
         model:get(to).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from).weight)
         model:get(to).bias[{{1,sz}}]:copy(pretrained:get(from).bias-0.25)
         model:get(to).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from).bias+0.25)
      end

      local function duplicateConv(from,to)

         local sz = pretrained:get(from).weight:size(2)
         model:get(to).weight[{{},{1,sz},{},{}}]:copy(pretrained:get(from).weight/2)
         model:get(to).weight[{{},{1+sz,sz*2},{},{}}]:copy(pretrained:get(from).weight/2)

      end

      model:get(1).weight:copy(pretrained:get(1).weight)
      duplicateBN(3,5)
      duplicateConv(6,8)
      duplicateBN(8,12)
      duplicateConv(11,15)
      duplicateBN(12,18)
      duplicateConv(15,21)
      duplicateBN(16,24)
      duplicateConv(19,27)
      duplicateBN(21,31)
      duplicateConv(24,34)
      duplicateBN(25,37)
      duplicateConv(29,41)
      model:remove(42)
      model:insert(pretrained:get(30),42)
      model:get(45).weight:copy(pretrained:get(33).weight)


   elseif opt.netType == 'resnet_sc_decoupled' then
      local function duplicateBN(from,from2,from3,from4,to,to2,to3,to4,opt)
         local sz = pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight:size(1)
         model:get(to):get(to2):get(to3):get(1):get(to4).running_mean[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).running_mean)
         model:get(to):get(to2):get(to3):get(1):get(to4).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).running_mean)
         model:get(to):get(to2):get(to3):get(1):get(to4).running_var[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).running_var)
         model:get(to):get(to2):get(to3):get(1):get(to4).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).running_var)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight)
         if opt == 'shift' then
            model:get(to):get(to2):get(to3):get(1):get(to4).bias[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).bias-0.25)
            model:get(to):get(to2):get(to3):get(1):get(to4).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).bias+0.25)
         else
            model:get(to):get(to2):get(to3):get(1):get(to4).bias[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).bias)
            model:get(to):get(to2):get(to3):get(1):get(to4).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).bias)
         end
      end

      local function duplicateBN_(from,to)
         local sz = pretrained:get(from).weight:size(1)
         model:get(to).running_mean[{{1,sz}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_var[{{1,sz}}]:copy(pretrained:get(from).running_var)
         model:get(to).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_var)
         model:get(to).weight[{{1,sz}}]:copy(pretrained:get(from).weight)
         model:get(to).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from).weight)
         model:get(to).bias[{{1,sz}}]:copy(pretrained:get(from).bias-0.25)
         model:get(to).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from).bias+0.25)
      end

      local function duplicateConv(from,from2,from3,from4,to,to2,to3,to4)
         local sz = pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight:size(2)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{},{1,sz},{},{}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight/2)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{},{1+sz,sz*2},{},{}}]:copy(pretrained:get(from):get(from2):get(from3):get(1):get(from4).weight/2)
      end

      model:get(1).weight:copy(pretrained:get(1).weight)
      model:remove(2)
      model:insert(pretrained:get(2),2)
      duplicateBN_(5,7)

      duplicateConv(6,1,1,3,8,1,1,3)
      duplicateConv(6,1,3,3,8,1,3,3)
      duplicateConv(6,2,1,3,8,2,1,3)
      duplicateConv(6,2,3,3,8,2,3,3)
      duplicateBN(6,1,1,4,8,1,1,6)
      duplicateBN(6,1,3,4,8,1,3,6)
      duplicateBN(6,2,1,4,8,2,1,6)
      duplicateBN(6,2,3,4,8,2,3,6)

      duplicateConv(7,1,1,3,9,1,1,3)
      duplicateConv(7,1,3,3,9,1,3,3)
      duplicateConv(7,2,1,3,9,2,1,3)
      duplicateConv(7,2,3,3,9,2,3,3)
      duplicateBN(7,1,1,4,9,1,1,6,'shift')
      duplicateBN(7,1,3,4,9,1,3,6)
      duplicateBN(7,2,1,4,9,2,1,6)
      duplicateBN(7,2,3,4,9,2,3,6)
      model:get(9):get(1):get(1):get(2):remove(4)
      model:get(9):get(1):get(1):get(2):insert(pretrained:get(7):get(1):get(1):get(2):get(2),4)
      model:get(9):get(1):get(1):get(2):remove(5)
      model:get(9):get(1):get(1):get(2):insert(pretrained:get(7):get(1):get(1):get(2):get(3),5)

      duplicateConv(8,1,1,3,10,1,1,3)
      duplicateConv(8,1,3,3,10,1,3,3)
      duplicateConv(8,2,1,3,10,2,1,3)
      duplicateConv(8,2,3,3,10,2,3,3)
      duplicateBN(8,1,1,4,10,1,1,6,'shift')
      duplicateBN(8,1,3,4,10,1,3,6)
      duplicateBN(8,2,1,4,10,2,1,6)
      duplicateBN(8,2,3,4,10,2,3,6)
      model:get(10):get(1):get(1):get(2):remove(4)
      model:get(10):get(1):get(1):get(2):insert(pretrained:get(8):get(1):get(1):get(2):get(2),4)
      model:get(10):get(1):get(1):get(2):remove(5)
      model:get(10):get(1):get(1):get(2):insert(pretrained:get(8):get(1):get(1):get(2):get(3),5)

      duplicateConv(9,1,1,3,11,1,1,3)
      duplicateConv(9,1,3,3,11,1,3,3)
      duplicateConv(9,2,1,3,11,2,1,3)
      duplicateConv(9,2,3,3,11,2,3,3)
      duplicateBN(9,1,1,4,11,1,1,6,'shift')
      duplicateBN(9,1,3,4,11,1,3,6)
      duplicateBN(9,2,1,4,11,2,1,6)
      -- duplicateBN(9,2,3,4,11,2,3,4)
      model:get(11):get(2):get(3):get(1):remove(4)
      model:get(11):get(2):get(3):get(1):insert(pretrained:get(9):get(2):get(3):get(1):get(4),4)
      model:get(11):get(1):get(1):get(2):remove(4)
      model:get(11):get(1):get(1):get(2):insert(pretrained:get(9):get(1):get(1):get(2):get(2),4)
      model:get(11):get(1):get(1):get(2):remove(5)
      model:get(11):get(1):get(1):get(2):insert(pretrained:get(9):get(1):get(1):get(2):get(3),5)

      model:remove(15)
      model:insert(pretrained:get(13),15)

   elseif opt.netType == 'resnet_decoupled' then
      local function duplicateBN(from,from2,from3,from4,to,to2,to3,to4,opt)
         local sz = pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight:size(1)
         model:get(to):get(to2):get(1):get(to3):get(to4).running_mean[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).running_mean)
         model:get(to):get(to2):get(1):get(to3):get(to4).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).running_mean)
         model:get(to):get(to2):get(1):get(to3):get(to4).running_var[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).running_var)
         model:get(to):get(to2):get(1):get(to3):get(to4).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).running_var)
         model:get(to):get(to2):get(1):get(to3):get(to4).weight[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight)
         model:get(to):get(to2):get(1):get(to3):get(to4).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight)
         if opt == 'shift' then
            model:get(to):get(to2):get(1):get(to3):get(to4).bias[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).bias-0.25)
            model:get(to):get(to2):get(1):get(to3):get(to4).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).bias+0.25)
         else
            model:get(to):get(to2):get(1):get(to3):get(to4).bias[{{1,sz}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).bias)
            model:get(to):get(to2):get(1):get(to3):get(to4).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).bias)
         end
      end

      local function duplicateBN_(from,to)
         local sz = pretrained:get(from).weight:size(1)
         model:get(to).running_mean[{{1,sz}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_mean[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_mean)
         model:get(to).running_var[{{1,sz}}]:copy(pretrained:get(from).running_var)
         model:get(to).running_var[{{1+sz,sz*2}}]:copy(pretrained:get(from).running_var)
         model:get(to).weight[{{1,sz}}]:copy(pretrained:get(from).weight)
         model:get(to).weight[{{1+sz,sz*2}}]:copy(pretrained:get(from).weight)
         model:get(to).bias[{{1,sz}}]:copy(pretrained:get(from).bias-0.25)
         model:get(to).bias[{{1+sz,sz*2}}]:copy(pretrained:get(from).bias+0.25)
      end

      local function duplicateConv(from,from2,from3,from4,to,to2,to3,to4)
         local sz = pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight:size(2)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{},{1,sz},{},{}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight/2)
         model:get(to):get(to2):get(to3):get(1):get(to4).weight[{{},{1+sz,sz*2},{},{}}]:copy(pretrained:get(from):get(from2):get(1):get(from3):get(from4).weight/2)
      end

      model:get(1).weight:copy(pretrained:get(1).weight)
      model:remove(2)
      model:insert(pretrained:get(2),2)
      duplicateBN_(5,7)

      duplicateConv(6,1,1,3,8,1,1,3)
      duplicateConv(6,1,1,7,8,1,1,9)
      duplicateConv(6,2,1,3,8,2,1,3)
      duplicateConv(6,2,1,7,8,2,1,9)
      duplicateBN(6,1,1,4,8,1,1,6,'shift')
      duplicateBN(6,1,1,8,8,1,1,12)
      duplicateBN(6,2,1,4,8,2,1,6,'shift')
      duplicateBN(6,2,1,8,8,2,1,12)

   
      duplicateConv(7,1,1,3,9,1,1,3)
      duplicateConv(7,1,1,7,9,1,1,9)
      duplicateConv(7,2,1,3,9,2,1,3)
      duplicateConv(7,2,1,7,9,2,1,9)
      duplicateBN(7,1,1,4,9,1,1,6,'shift')
      duplicateBN(7,1,1,8,9,1,1,12,'shift')
      duplicateBN(7,2,1,4,9,2,1,6,'shift')
      duplicateBN(7,2,1,8,9,2,1,12)
      model:get(9):get(1):get(1):get(2):remove(4)
      model:get(9):get(1):get(1):get(2):insert(pretrained:get(7):get(1):get(1):get(2):get(2),4)
      model:get(9):get(1):get(1):get(2):remove(5)
      model:get(9):get(1):get(1):get(2):insert(pretrained:get(7):get(1):get(1):get(2):get(3),5)

      duplicateConv(8,1,1,3,10,1,1,3)
      duplicateConv(8,1,1,7,10,1,1,9)
      duplicateConv(8,2,1,3,10,2,1,3)
      duplicateConv(8,2,1,7,10,2,1,9)
      duplicateBN(8,1,1,4,10,1,1,6,'shift')
      duplicateBN(8,1,1,8,10,1,1,12,'shift')
      duplicateBN(8,2,1,4,10,2,1,6,'shift')
      duplicateBN(8,2,1,8,10,2,1,12)
      model:get(10):get(1):get(1):get(2):remove(4)
      model:get(10):get(1):get(1):get(2):insert(pretrained:get(8):get(1):get(1):get(2):get(2),4)
      model:get(10):get(1):get(1):get(2):remove(5)
      model:get(10):get(1):get(1):get(2):insert(pretrained:get(8):get(1):get(1):get(2):get(3),5)

      duplicateConv(9,1,1,3,11,1,1,3)
      duplicateConv(9,1,1,7,11,1,1,9)
      duplicateConv(9,2,1,3,11,2,1,3)
      duplicateConv(9,2,1,7,11,2,1,9)
      duplicateBN(9,1,1,4,11,1,1,6,'shift')
      duplicateBN(9,1,1,8,11,1,1,12,'shift')
      duplicateBN(9,2,1,4,11,2,1,6,'shift')
      -- duplicateBN(8,2,1,8,10,2,1,12)
      model:get(11):get(2):get(1):get(1):remove(10)
      model:get(11):get(2):get(1):get(1):insert(pretrained:get(9):get(2):get(1):get(1):get(8),10)
      model:get(11):get(1):get(1):get(2):remove(4)
      model:get(11):get(1):get(1):get(2):insert(pretrained:get(9):get(1):get(1):get(2):get(2),4)
      model:get(11):get(1):get(1):get(2):remove(5)
      model:get(11):get(1):get(1):get(2):insert(pretrained:get(9):get(1):get(1):get(2):get(3),5)

      model:remove(15)
      model:insert(pretrained:get(13),15)

   end
   return model
end

return M
