--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.

local nn = require 'nn'
require 'cunn'
require '../newLayers/QuantizedNeurons.lua'
require '../newLayers/cudnnQuantSpatialConvolution.lua'

local Convolution = cudnn.SpatialConvolution
local ConvolutionQ = cudnnQuantSpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride, type)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane and type ~= 'last')
      if useConv then
         -- 1x1 convolution
         local f=nn.Sequential()
         if nInputPlane == 64 then
            f:add(nn.Reshape(2,nInputPlane,56,56,true)) 
         elseif nInputPlane == 90 then
            f:add(nn.Reshape(2,nInputPlane,28,28,true))
         elseif nInputPlane == 180 then
            f:add(nn.Reshape(2,nInputPlane,14,14,true))
         elseif nInputPlane == 360 then
            f:add(nn.Reshape(2,nInputPlane,7,7,true))
         else
            error('not case')
         end
         f:add(nn.Mean(1,4))
         f:add(nn.SpatialAveragePooling(2,2,2,2))
         f:add(Convolution(nInputPlane, nOutputPlane, 1, 1, 1, 1))
         f:add(SBatchNorm(nOutputPlane))
         f:add(nn.Replicate(2,1,3))                  
         if nOutputPlane == 64 then
            f:add(nn.Reshape(nOutputPlane*2,56,56,true)) 
         elseif nOutputPlane == 90 then
            f:add(nn.Reshape(nOutputPlane*2,28,28,true))
         elseif nOutputPlane == 180 then
            f:add(nn.Reshape(nOutputPlane*2,14,14,true))
         elseif nOutputPlane == 360 or nOutputPlane == 512 then
            f:add(nn.Reshape(nOutputPlane*2,7,7,true))
         else
            error('not case')
         end
         return f
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.Reshape(2,360,7,7,true))
            :add(nn.Mean(1,4))
            -- :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.Sequential()
                  :add(nn.MulConstant(0))
                  :add(nn.Narrow(2,1,152))))
      else
         return nn.Identity()
      end
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end


   local function residual(nInputPlane, nOutputPlane, stride,type)
      local s = nn.Sequential()

      if opt.activation == 'ReLU' then
         s:add(ReLU(true))
      elseif opt.activation == 'sign' then
         s:add(nn.HardTanh(-1,1,false))
         s:add(QuantizedNeurons(1,1))      
      elseif opt.activation == 'ClippedReLU1' then
         s:add(nn.HardTanh(0,1,false))
         s:add(QuantizedNeurons(opt.Abits,0))
      else
         error('The activation function type is not supported.'+opt.activation)
      end
      s:add(ConvolutionQ(nInputPlane*2,nOutputPlane,3,3,stride,stride,1,1,opt.Wbits,opt.clipW))
      if type == 'last' then
         s:add(SBatchNorm(nOutputPlane))
      else
         s:add(nn.Replicate(2,1,3))                  
         if nOutputPlane == 64 then
            s:add(nn.Reshape(nOutputPlane*2,56,56,true)) 
         elseif nOutputPlane == 90 then
            s:add(nn.Reshape(nOutputPlane*2,28,28,true))
         elseif nOutputPlane == 180 then
            s:add(nn.Reshape(nOutputPlane*2,14,14,true))
         elseif nOutputPlane == 360 or nOutputPlane == 512 then
            s:add(nn.Reshape(nOutputPlane*2,7,7,true))
         else
            print(nInputPlane)
            print(nOutputPlane)
            error('not case')
         end
         s:add(SBatchNorm(nOutputPlane*2))
      end
      return s
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n

      local block = nn.Sequential()
      if type == '512' then
         return block
         :add(nn.ConcatTable()
            :add(residual(nInputPlane,n,stride,type))
            :add(shortcut(nInputPlane,n,stride,type)))
         :add(nn.CAddTable(true))
         :add(nn.ConcatTable()
            :add(residual(n, 512, 1,'last'))
            :add(shortcut(n, 512, 1,'last')))
         :add(nn.CAddTable(true))
      else
         return block
         :add(nn.ConcatTable()
            :add(residual(nInputPlane,n,stride,type))
            :add(shortcut(nInputPlane,n,stride,type)))
         :add(nn.CAddTable(true))
         :add(nn.ConcatTable()
            :add(residual(n, n, 1))
            :add(shortcut(n, n, 1)))
         :add(nn.CAddTable(true))
      end   
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n * 4

      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))

      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride, type)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      s:add(block(features, stride))
      for i=2,count do
         s:add(block(features, 1,type == 'last' and '512' or ''))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
         [200] = {{3, 24, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model

      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(nn.Replicate(2,1,3))                  
      model:add(nn.Reshape(64*2,56,56,true)) 
      model:add(SBatchNorm(128))
      model:add(layer(block, 64,  def[1], 1, 'first'))
      model:add(layer(block, 90, def[2], 2))
      model:add(layer(block, 180, def[3], 2))
      model:add(layer(block, 360, def[4], 2, 'last'))
      model:add(ReLU(true))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(layer(basicblock, 16, n, 1))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
      model:add(ReLU(true))
      -- model:add(PACT())
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(layer(basicblock, 16, n, 1))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nInputPlane + torch.ceil(v.kW/v.dW)*torch.ceil(v.kH/v.dH)*v.nOutputPlane
         
         v.weight:normal(0,math.sqrt(4/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnnQuantSpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel