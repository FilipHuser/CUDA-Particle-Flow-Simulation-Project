#pragma once

#include<type_traits>

namespace pg
{

	//This template will be specialized later with custom enum type representing a bit mask
	template<typename E>
	struct enable_bitmask_operators
	{
		static const bool enable=false;
	};

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E> operator|(const E lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		return static_cast<E>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E> operator&(const E lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		return static_cast<E>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E> operator^(const E lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		return static_cast<E>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E> operator~(const E lhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		return static_cast<E>(~static_cast<underlying>(lhs));
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E&> operator|=(E& lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		lhs=static_cast<E>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
		return lhs;
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E&> operator&=(E& lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		lhs=static_cast<E>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
		return lhs;
	}

	template<typename E>
	constexpr typename std::enable_if_t<enable_bitmask_operators<E>::enable,E&> operator^=(E& lhs, const E rhs)
	{
		typedef typename std::underlying_type<E>::type underlying;
		lhs=static_cast<E>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
		return lhs;
	}

}