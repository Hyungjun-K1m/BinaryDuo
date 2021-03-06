local nn = require 'nn'
require 'cunn'
require '../newLayers/QuantizedNeurons.lua'


local Convolution = cudnn.SpatialConvolution
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
            modell:add(QuantizedNeurons(1,0))
         else
            error('activation function not supported!')
         end
      end 
   end

   -- The tiny VGG-7 CIFAR-10 model
   local model = nn.Sequential()
   model:add(Convolution(3,45,3,3,1,1,1,1))     
   model:add(nn.Replicate(2,1,3))               
   model:add(nn.Reshape(90,32,32,true))         
   model:add(SBatchNorm(90))                    
   add_activation(model,opt.activation,1)       
   
   model:add(Convolution(90,45,3,3,1,1,1,1))    
   model:add(Max(2,2,2,2))                      
   model:add(nn.Replicate(2,1,3))               
   model:add(nn.Reshape(90,16,16,true))         
   model:add(SBatchNorm(90))                    
   add_activation(model,opt.activation,1)       
   
   model:add(Convolution(90,90,3,3,1,1,1,1))    
   model:add(Max(2,2,2,2))                      
   model:add(nn.Replicate(2,1,3))               
   model:add(nn.Reshape(180,8,8,true))          
   model:add(SBatchNorm(180))                   
   add_activation(model,opt.activation,1)       

   model:add(Convolution(180,90,3,3,1,1,1,1))   
   model:add(Max(2,2,2,2))                      
   model:add(nn.Replicate(2,1,3))               
   model:add(nn.Reshape(180,4,4,true))          
   model:add(SBatchNorm(180))                   
   add_activation(model,opt.activation,1)       
   
   model:add(Convolution(180,336,4,4,4,4,0,0))  
   model:add(nn.Replicate(2,1,3))               
   model:add(nn.Reshape(672,1,1,true))          
   model:add(SBatchNorm(672))                   
   add_activation(model,opt.activation,1)       
   
   model:add(Convolution(672,512,1,1,1,1,0,0))  
   model:add(SBatchNorm(512))                   
   model:add(ReLU(true))                        
   model:add(Convolution(512,10,1,1,1,1,0,0))   
   model:add(nn.View(10))                       
   
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = 0
         if v.__typename == 'cudnn.SpatialConvolution' then
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