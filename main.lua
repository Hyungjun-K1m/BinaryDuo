--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local opts = require 'opts'

local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
opt = opts.parse(arg)

local param_logger = io.open(paths.concat(opt.save, 'params.t7'),'a')
for param,val in pairs(opt) do
   param_logger:write(param..'\t'..tostring(val)..'\n')
end
param_logger:close()

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
local Trainer = require 'train'

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)
print(model)
if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   if opt.pretraining then
      checkpoints.save(0,model,trainer.optimState, false, opt)
   end
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
bestTop1, bestTop5, bestTrainTop1 = math.huge, math.huge, math.huge

local timerT = torch.Timer()  
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   collectgarbage()
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
   if trainTop1 < bestTrainTop1 then
      bestTrainTop1 = trainTop1
   end
   collectgarbage()
   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)
   collectgarbage()
   bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
   end
   if testTop5 < bestTop5 then
      bestTop5 = testTop5
   end
   if opt.dataset == 'imagenet' then
      if epoch %10 == 0 or bestModel then
         checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
      end
   elseif opt.dataset == 'cifar10' then
      if opt.pretraining and (epoch == 100 or (epoch > 100 and bestModel)) then
         checkpoints.save(epoch,model,trainer.optimState, bestModel, opt)
      end
   end
   print('Epoch Time: ',timerT:time().real)
   timerT:reset()
end