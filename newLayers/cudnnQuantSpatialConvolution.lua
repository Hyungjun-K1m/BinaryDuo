-- require 'cudnn'
local cudnnQuantSpatialConvolution, parent =
    torch.class('cudnnQuantSpatialConvolution', 'cudnn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

function cudnnQuantSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, Wbits, clip, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    self.Wbits = Wbits or false
    self.clip = clip or 0
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.weightQ = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self:noBias()
end


function cudnnQuantSpatialConvolution:binarized(trainFlag)
  local absE = self.weight:abs():mean()
  self.weightQ:copy(self.weightOrg):sign():add(0.1):sign():mul(absE)
  return self.weightQ
end


-- if you change the configuration of the module manually, call this
function cudnnQuantSpatialConvolution:resetWeightDescriptors()
    assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- for compatibility
    self.groups = self.groups or 1
    -- create filterDescriptor for weight

    self.weightDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.weight)],
         filterDimA = desc or
            {self.nOutputPlane/self.groups,
             self.nInputPlane/self.groups,
             self.kH, self.kW}
       }
    )

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end

    return self

end

function cudnnQuantSpatialConvolution:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
      self.fastest_mode = mode
      self.iDesc = nil
    end
    return self
end

function cudnnQuantSpatialConvolution:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    -- self.iSize = self.iSize or torch.LongStorage(4)
    -- self.iSize:fill(0)
    return self
end

function cudnnQuantSpatialConvolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function cudnnQuantSpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function cudnnQuantSpatialConvolution:checkInputChanged(input)
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.groups = self.groups or 1
    if not self.weightDesc then self:resetWeightDescriptors() end
    if not self.weightDesc then error "Weights not assigned!" end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert(self.nInputPlane == input:size(2),
              'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
       return true
    end
    return false
end

function cudnnQuantSpatialConvolution:createIODescriptors(input)
    parent.createIODescriptors(self,input)
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function cudnnQuantSpatialConvolution:updateOutput(input)
    if self.clip ~= 0 then
      self.weight:clamp(-self.clip,self.clip)
    end
    self.weightOrg:copy(self.weight)
    if self.Wbits == 0 then
      self.weightQ:copy(self.weightOrg)
    elseif self.Wbits == 1 then
      self.weightQ = self:binarized(self.train)
    else
      error('weight quantization to other than 1-bit is not supported.')
    end

    self.weight:copy(self.weightQ)
    parent.updateOutput(self,input)
    self.weight:copy(self.weightOrg)
    return self.output
end

function cudnnQuantSpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.weight:copy(self.weightQ)
    parent.updateGradInput(self, input, gradOutput:contiguous(), scale)
    self.weight:copy(self.weightOrg)
    return self.gradInput
end

function cudnnQuantSpatialConvolution:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput:contiguous(), scale)
end

function cudnnQuantSpatialConvolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function cudnnQuantSpatialConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function cudnnQuantSpatialConvolution:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end