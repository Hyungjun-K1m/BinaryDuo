--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '/media/ramdisk/images/',         'Path to dataset')
   cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-precision', 'single',    'Options: single | double | half')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        8, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       256,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-run',            1,          'number of models to be trained simultaneously.')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   cmd:option('-deploy',          'none',          'Deploy pre-trained weights')
   cmd:option('-deployOpt',        0,              'deployment option. 0~6')
   cmd:option('-pretraining',     false,           'If true, the model will be saved at the last epoch to be used as pre-trained model.')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.1,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-dropout',         0.1,    'dropout ratio')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   cmd:option('-optimType',       'sgd',  'optimizer type: sgd | adam | adamax')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet', 'Options: resnet | preresnet')
   cmd:option('-activation',     'ClippedReLU1',  'Activation function to use')
   cmd:option('-Wbits',         0,        'Weight resolution')
   cmd:option('-Abits',         0,        'Activation resolution')
   cmd:option('-depth',         0,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-clipW',          0,       'weight clipping parameter. if 0, no clipping, otherwise clip the weight with -clipW/clipW.')
   cmd:option('-shortcutType', 'B',       'Options: A | B | C')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:option('-note',            '',        'any note for further information')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   -- add date/time
   opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''):gsub(':','-'))
   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end
   print('saving everything to '.. opt.save)

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      if opt.depth == 0 then
         opt.nEpochs = opt.nEpochs == 0 and 60 or opt.nEpochs
      else
         opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
         opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
      end
   elseif opt.dataset == 'cifar10' then
      if opt.depth == 0 then
         opt.nEpochs = opt.nEpochs == 0 and 120 or opt.nEpochs
      else
         opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
         opt.nEpochs = opt.nEpochs == 0 and 100 or opt.nEpochs
      end
   elseif opt.dataset == 'cifar100' then
       -- Default shortcutType=A and nEpochs=164
       opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
       opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end
   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
