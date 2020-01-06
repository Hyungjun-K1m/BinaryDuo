local nn = require 'nn'
require 'cunn'
require '../newLayers/QuantizedNeurons.lua'
require '../newLayers/cudnnQuantSpatialConvolution.lua'


local Convolution = cudnn.SpatialConvolution
local ConvolutionQ = cudnnQuantSpatialConvolution
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization


local function createModel(opt)

   local function add_activation(modell,Atype,Abits)
      if Abits == 0 then
         modell:add(ReLU(true))
      else
         if Atype == 'ClippedReLU1' then
            modell:add(nn.HardTanh(0,1,true))
            modell:add(QuantizedNeurons(0.3,0))
         else
            error('activation function not supported!')
         end
      end 
   end

   
   local model = nn.Sequential()
   model:add(Convolution(3,88,11,11,4,4,2,2))
   model:add(Max(3,3,2,2)) 
   model:add(SBatchNorm(88))
   add_activation(model,opt.activation,opt.Abits)
   
   model:add(ConvolutionQ(88,192,5,5,1,1,2,2,opt.Wbits,opt.clipW))
   model:add(Max(3,3,2,2))
   model:add(SBatchNorm(192))
   add_activation(model,opt.activation,opt.Abits)
   
   model:add(ConvolutionQ(192,256,3,3,1,1,1,1,opt.Wbits,opt.clipW))
   model:add(SBatchNorm(256))
   add_activation(model,opt.activation,opt.Abits)

   model:add(ConvolutionQ(256,256,3,3,1,1,1,1,opt.Wbits,opt.clipW))
   model:add(SBatchNorm(256))
   add_activation(model,opt.activation,opt.Abits)
   
   model:add(ConvolutionQ(256,192,3,3,1,1,1,1,opt.Wbits,opt.clipW))
   model:add(Max(3,3,2,2))
   model:add(SBatchNorm(192))
   add_activation(model,opt.activation,opt.Abits)

   model:add(ConvolutionQ(192,2400,6,6,6,6,0,0,opt.Wbits,opt.clipW))
   model:add(SBatchNorm(2400))
   add_activation(model,opt.activation,opt.Abits)
   model:add(nn.SpatialDropout(opt.dropout))
   
   model:add(ConvolutionQ(2400,4096,1,1,1,1,0,0,opt.Wbits,opt.clipW))
   model:add(SBatchNorm(4096))
   model:add(ReLU(true))
   model:add(nn.SpatialDropout(opt.dropout))
   model:add(Convolution(4096,1000,1,1,1,1,0,0))
   model:add(nn.View(1000))
   
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = 0
         if v.__typename == 'cudnn.SpatialConvolution' or v.__typename == 'cudnnQuantSpatialConvolution' then
            print('init '..v.__typename)
            n = v.kW*v.kH*v.nInputPlane + torch.ceil(v.kW/v.dW)*torch.ceil(v.kH/v.dH)*v.nOutputPlane
         end
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
   ConvInit('nn.SpatialConvolution')
   ConvInit('cudnnQuantSpatialConvolution')
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

   -- model:get(1).gradInput = nil

   return model
end

return createModel