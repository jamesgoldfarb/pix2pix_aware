--[[
    NIfTI HU data loader for paired CT volumes.

    Expects directory layout:
      $DATA_ROOT/A/<phase>/*.nii or *.nii.gz
      $DATA_ROOT/B/<phase>/*.nii or *.nii.gz

    Each A file must have a matching B file with identical filename in the B tree.
    This loader samples axial slices, applies optional exclude_slices at both volume ends,
    resizes/crops, converts HU linearly to [-1,1], and concatenates A and B as channels.
]]--

require 'image'
local ok, nifti = pcall(require, 'nifti')
assert(ok, 'Missing LuaRocks module "nifti". Install with: luarocks install nifti')

paths.dofile('dataset.lua')
util = paths.dofile('util/util.lua')

-- Resolve data roots
opt.dataA = paths.concat(os.getenv('DATA_ROOT'), 'A', opt.phase)
opt.dataB = paths.concat(os.getenv('DATA_ROOT'), 'B', opt.phase)
opt.data  = opt.dataA

if not paths.dirp(opt.dataA) then error('Did not find directory: ' .. opt.dataA) end
if not paths.dirp(opt.dataB) then error('Did not find directory: ' .. opt.dataB) end

-- Cache filename
local cache = 'cache'
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p ' .. cache)
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

-- Parameters
-- Expect single-channel HU inputs/outputs; enforce via options
local input_nc = tonumber(opt.input_nc or 1)
local output_nc = tonumber(opt.output_nc or 1)
assert(input_nc == 1 and output_nc == 1, 'nifti_hu mode requires input_nc=1 and output_nc=1')
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local hu_min = tonumber(opt.hu_min or -1000)
local hu_max = tonumber(opt.hu_max or  3000)
local exclude_slices = tonumber(opt.exclude_slices or 0)

local function read_volume(path)
  local obj = nifti.load(path)
  local vol = obj.getTensor and obj:getTensor() or obj.data
  assert(vol, 'Failed to read NIfTI volume at ' .. path)
  return vol:float()
end

local function pick_slice_idx(depth)
  local start_idx = 1 + math.max(0, exclude_slices)
  local end_idx = depth - math.max(0, exclude_slices)
  if end_idx < start_idx then
    -- fallback to center slice
    return math.max(1, math.ceil(depth/2))
  end
  if opt.phase == 'train' and opt.serial_batches ~= 1 then
    return torch.random(start_idx, end_idx)
  else
    return math.floor((start_idx + end_idx) / 2)
  end
end

local function prepare_slice(vol, slice_idx)
  -- Expect vol shape HxWxZ or HxW
  if vol:nDimension() == 2 then
    return vol
  elseif vol:nDimension() == 3 then
    return vol[{{}, {}, {slice_idx}}]:squeeze()
  elseif vol:nDimension() == 4 then
    -- [H,W,Z,T] -> take T=1
    return vol[{{}, {}, {slice_idx}, {1}}]:squeeze()
  else
    error('Unsupported volume dims: ' .. tostring(vol:nDimension()))
  end
end

local function resize_and_crop2d(im)
  -- im: HxW float single-channel
  im = image.scale(im, loadSize[2], loadSize[2])
  local o = opt.fineSize
  local iH, iW = im:size(1), im:size(2)
  local h1, w1 = 1, 1
  if opt.phase == 'train' then
    if iH ~= o then h1 = math.ceil(torch.uniform(1e-2, iH - o)) end
    if iW ~= o then w1 = math.ceil(torch.uniform(1e-2, iW - o)) end
  else
    h1 = math.ceil((iH - o) / 2)
    w1 = math.ceil((iW - o) / 2)
  end
  im = image.crop(im, w1, h1, w1 + o, h1 + o)
  if opt.flip == 1 and opt.phase == 'train' and torch.uniform() > 0.5 then
    im = image.hflip(im)
  end
  return im
end

local function sample_pair(pathA)
  collectgarbage()
  local pathB = pathA:gsub('/A/', '/B/')
  assert(paths.filep(pathB), 'Missing paired volume for ' .. pathA)

  local volA = read_volume(pathA)
  local volB = read_volume(pathB)

  -- match depth
  local zA = (volA:nDimension()==3 and volA:size(3)) or (volA:nDimension()==4 and volA:size(3)) or 1
  local zB = (volB:nDimension()==3 and volB:size(3)) or (volB:nDimension()==4 and volB:size(3)) or 1
  local z = math.min(zA, zB)
  local idx = pick_slice_idx(z)

  local slA = prepare_slice(volA, math.min(idx, zA))
  local slB = prepare_slice(volB, math.min(idx, zB))

  slA = resize_and_crop2d(slA)
  slB = resize_and_crop2d(slB)

  slA = util.preprocess_HU(slA, hu_min, hu_max)
  slB = util.preprocess_HU(slB, hu_min, hu_max)

  local imA = slA:view(1, slA:size(1), slA:size(2))
  local imB = slB:view(1, slB:size(1), slB:size(2))
  local imAB = torch.cat(imA, imB, 1)
  assert(imAB:max() <= 1 and imAB:min() >= -1, 'badly scaled HU input')
  return imAB
end

-- Hooks
local function trainHook(self, pathA)
  return sample_pair(pathA)
end
local function testHook(self, pathA)
  return sample_pair(pathA)
end

-- Create loader over A paths; B derived by gsub in hook
trainLoader = dataLoader{
  paths = {opt.data},
  loadSize = {input_nc, loadSize[2], loadSize[2]},
  sampleSize = {input_nc + output_nc, sampleSize[2], sampleSize[2]},
  split = 100,
  serial_batches = opt.serial_batches,
  verbose = true
}

trainLoader.sampleHookTrain = trainHook
trainLoader.sampleHookTest = testHook

-- sanity checks
local class = trainLoader.imageClass
local nClasses = #trainLoader.classes
assert(class:max() <= nClasses, 'class logic has error')
assert(class:min() >= 1, 'class logic has error')
