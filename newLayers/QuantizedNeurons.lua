local QuantizedNeurons,parent = torch.class('QuantizedNeurons', 'nn.Module')


function QuantizedNeurons:__init(bitA,binarymode,exception)
  parent.__init(self)
  if bitA > 0.9999 then
    self.n = 2^bitA - 1
  elseif bitA == 0 then
    self.n = 0
  else
    self.n = bitA*10-1
  end
  -- print(self.n)
  self.mode = binarymode or 0
  self.exception = exception or false
end

function QuantizedNeurons:updateOutput(input)

  self.output:resizeAs(input)
  self.output:copy(input)
  if self.mode == 1 then
    -- -1/+1 mode
    self.output:add(1):mul(0.5)
  end

  -- if bitA == 0, it means no quantization
  if self.n ~= 0 then
    if self.exception then
      local sz = input:size()
      self.output[{{},{1,sz[2]/2},{},{}}]:add(0.25)
      self.output[{{},{1+sz[2]/2,sz[2]},{},{}}]:add(-0.25)
    end
    self.output:mul(self.n):round():div(self.n)
  end
  if self.mode == 1 then
    -- -1/+1 mode
    self.output:mul(2):add(-1)
  end
 return self.output
end

function QuantizedNeurons:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
end