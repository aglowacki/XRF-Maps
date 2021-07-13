/***
Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.

Copyright 2016. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***/




#include "stream_block_allocator.h"
#include <thread>

namespace data_struct
{

std::mutex Stream_Block_Allocator::_mutex;
Stream_Block_Allocator* Stream_Block_Allocator::_this_inst(nullptr);

//-----------------------------------------------------------------------------

Stream_Block_Allocator::Stream_Block_Allocator()
{
    _mem_limit = -1;
    _cur_mem_usage = 0;
    _clean_up = false;
}

//-----------------------------------------------------------------------------

Stream_Block_Allocator::~Stream_Block_Allocator()
{
    clean_up_all_stream_blocks();
}

//-----------------------------------------------------------------------------

Stream_Block_Allocator* Stream_Block_Allocator::inst()
{
    std::lock_guard<std::mutex> lock(_mutex);

    if (_this_inst == nullptr)
    {
        _this_inst = new Stream_Block_Allocator();
    }
    return _this_inst;
}

//-----------------------------------------------------------------------------s

void Stream_Block_Allocator::set_mem_limit(long long limit)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _mem_limit = limit;
}

//-----------------------------------------------------------------------------

data_struct::Stream_Block* Stream_Block_Allocator::alloc_stream_block(int detector, size_t row, size_t col, size_t height, size_t width, size_t spectra_size)
{
    long long extra = ( ((spectra_size + 4) * sizeof(real_t)) + ((11 * sizeof(size_t)) + 4) ); // size of stream_block

    long long cur = _cur_mem_usage.load(std::memory_order_acquire);

    if (cur >= _mem_limit)
    {
        while (_clean_up == false)
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_free_stream_blocks.size() > 0)
                {
                    Stream_Block* sb = _free_stream_blocks.back();
                    _free_stream_blocks.pop_back();
                    _stream_blocks.push_back(sb);
                    return sb;
                }
            }

            std::this_thread::yield();
        }
    }

    /*
    while (_cur_mem_usage.load(std::memory_order_acquire) >= (_mem_limit + extra))
    {
        std::this_thread::yield();
    }
    */
    if (_clean_up == false)
    {

        std::lock_guard<std::mutex> lock(_mutex);

        _stream_blocks.push_back(new Stream_Block(detector, row, col, height, width, spectra_size));
        _cur_mem_usage.store(cur+extra, std::memory_order_release);
        return _stream_blocks.back();
    }
    return nullptr;
}

//-----------------------------------------------------------------------------

void Stream_Block_Allocator::free_stream_blocks(data_struct::Stream_Block* sb)
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    for (std::list<Stream_Block*>::iterator itr = _stream_blocks.begin(); itr != _stream_blocks.end(); ++itr)
    {
        if (sb == *itr)
        {
            _stream_blocks.erase(itr);
            break;
        }
    }
    
    _free_stream_blocks.push_back(sb);
    
    /*
    std::vector<Stream_Block*>::iterator itr = _stream_blocks.begin();

    while (itr != _stream_blocks.end())
    {
        if ((*itr) == sb)
        {
            //long long cur = _cur_mem_usage.load(std::memory_order_acquire);
            //long long sub = cur - ( ((*itr)->samples() * sizeof(real_t)) + (5 * sizeof(size_t)));
            //delete (*itr);
            //_cur_mem_usage.store(sub, std::memory_order_release);
            itr = _stream_blocks.erase(itr);
            _free_stream_blocks.push_back(*itr);
            break;
        }
        else
        {
            ++itr;
        }
    }
    */
}

//-----------------------------------------------------------------------------

void Stream_Block_Allocator::clean_up_all_stream_blocks()
{
    std::lock_guard<std::mutex> lock(_mutex);

    _clean_up = true;

    for (auto& itr : _free_stream_blocks)
    {
        delete itr;
    }

    for(auto &itr: _stream_blocks)
    {
        delete itr;
    }

    _cur_mem_usage.store(0, std::memory_order_release);
}

//-----------------------------------------------------------------------------

} //namespace data_struct
